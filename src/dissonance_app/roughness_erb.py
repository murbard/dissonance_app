import torch
import torch.nn.functional as F
import torchaudio.functional as TAF
import numpy as np
import scipy.signal
import math
from dissonance_app.differentiable_filters import diff_butter_bandpass_sos, diff_butter_lowpass_sos

# ---------- ERB spacing helpers (PyTorch versions) ----------

def hz_to_erbrate(f_hz: torch.Tensor) -> torch.Tensor:
    return 21.4 * torch.log10(4.37e-3 * f_hz + 1.0)

def erbrate_to_hz(erbrate: torch.Tensor) -> torch.Tensor:
    return (torch.pow(10, erbrate / 21.4) - 1.0) / 4.37e-3

def erb_bw_hz(f_hz: torch.Tensor) -> torch.Tensor:
    # ERB(f) ≈ 24.7 * (4.37*f/1000 + 1)   [Hz]
    return 24.7 * (4.37e-3 * f_hz + 1.0)

def erb_spaced_center_freqs(fmin_hz: float, fmax_hz: float, n_bands: int, device=None) -> torch.Tensor:
    if device is None:
        device = torch.device('cpu')
    
    # Use numpy for linspace logic to match exactly, then convert
    # or implement in torch
    fmin = torch.tensor(fmin_hz, device=device)
    fmax = torch.tensor(fmax_hz, device=device)
    
    erbs = torch.linspace(hz_to_erbrate(fmin).item(), hz_to_erbrate(fmax).item(), n_bands, device=device)
    return erbrate_to_hz(erbs)


# ---------- small utilities ----------

def moving_rms(x: torch.Tensor, win_samples: int) -> torch.Tensor:
    """
    Compute moving RMS using 1D convolution.
    x: (batch, time) or (time,)
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    
    # x shape: (B, T)
    if win_samples <= 1:
        return torch.abs(x)
        
    # Kernel for moving average
    kernel = torch.ones(1, 1, win_samples, device=x.device) / win_samples
    
    # Padding "same" logic
    # Conv1d output size L_out = L_in + 2*padding - dilation*(kernel_size-1) - 1 + 1
    # We want L_out = L_in.
    # padding = (win_samples - 1) / 2
    pad_total = win_samples - 1
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    
    x_squared = x * x
    # Add channel dim for conv1d: (B, C, T) -> (B, 1, T)
    x_squared_padded = F.pad(x_squared.unsqueeze(1), (pad_left, pad_right))
    
    power = F.conv1d(x_squared_padded, kernel)
    
    # Remove channel dim
    power = power.squeeze(1)
    
    return torch.sqrt(torch.clamp(power, min=0.0))

def hilbert_envelope(x: torch.Tensor) -> torch.Tensor:
    """
    Compute envelope via Hilbert transform (fft -> mask -> ifft).
    x: (B, T)
    """
    n = x.shape[-1]
    Xf = torch.fft.fft(x, n=n, dim=-1)
    
    h = torch.zeros(n, device=x.device)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
        
    X_analytic = Xf * h
    analytic_signal = torch.fft.ifft(X_analytic, dim=-1)
    return torch.abs(analytic_signal)


# ---------- main: roughness curve ----------

def roughness_curve_erb(
    x: torch.Tensor,
    fs: float,
    *,
    n_bands: int = 64,
    fmin: float = 50.0,
    fmax: float = None,
    bandpass_order: int = 4,
    erb_bw_scale: float = 1.47,
    envelope_method: str = "hilbert",
    env_lp_hz: float = 400.0,
    mod_band_hz: tuple = (13.0, 120.0),
    rough_rms_win_ms: float = 20.0,
    level_lp_hz: float = 3.0,
    level_compression: float = 0.66,
    output_hop_s: float = 0.005,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PyTorch implementation of roughness_curve_filterbank.
    
    Returns:
        t_out: times (seconds)
        R_out: roughness curve
        centers: center freqs
    """
    device = x.device
    device = x.device
    # Implicitly handle (Batch, Time) or (Time)
    if x.dim() == 1:
        x = x.unsqueeze(0) # (1, T)
        
    # Normalize to -1, 1 if not already (assuming input is float32)
    
    nyq = 0.5 * fs
    if fmax is None:
        fmax = min(12000.0, 0.95 * nyq)
    fmax = float(min(fmax, 0.95 * nyq))
    fmin = float(max(fmin, 1.0))
    
    centers = erb_spaced_center_freqs(fmin, fmax, n_bands, device=device)
    
    # Design SOS Filters 
    # Logic: if params are tensors requiring grad, use differentiable design.
    # Otherwise use SciPy (cached/fast).
    
    def use_differentiable():
        return (torch.is_tensor(erb_bw_scale) and erb_bw_scale.requires_grad) or \
               (torch.is_tensor(mod_band_hz) and mod_band_hz.requires_grad) or \
               (torch.is_tensor(level_compression) and level_compression.requires_grad) or \
               isinstance(mod_band_hz, tuple) and any(torch.is_tensor(x) and x.requires_grad for x in mod_band_hz)

    # Helper to standardize mod_band_hz
    lo_mod, hi_mod = mod_band_hz
    if not torch.is_tensor(lo_mod): lo_mod = torch.tensor(lo_mod, device=device)
    if not torch.is_tensor(hi_mod): hi_mod = torch.tensor(hi_mod, device=device)

    # Modulation Filter
    # Use SciPy for inference (accurate); differentiable for training (gradient-enabled).
    def design_butter_sos_scipy(order, freqs, btype):
        sos_np = scipy.signal.butter(order, freqs, btype=btype, fs=fs, output='sos')
        return torch.from_numpy(sos_np).to(device=device, dtype=x.dtype)
    
    needs_grad = (torch.is_tensor(lo_mod) and lo_mod.requires_grad) or \
                 (torch.is_tensor(hi_mod) and hi_mod.requires_grad) or \
                 (torch.is_tensor(erb_bw_scale) and erb_bw_scale.requires_grad) or \
                 (torch.is_tensor(level_compression) and level_compression.requires_grad)
    
    if needs_grad:
        sos_mod = diff_butter_bandpass_sos(lo_mod, hi_mod, fs, order=4).to(device)
        sos_level = diff_butter_lowpass_sos(level_lp_hz, fs, order=2).to(device)
        sos_env = None
        if envelope_method == "rectify_lp":
             sos_env = diff_butter_lowpass_sos(env_lp_hz, fs, order=2).to(device)
    else:
        lo_val = lo_mod.item() if torch.is_tensor(lo_mod) else lo_mod
        hi_val = hi_mod.item() if torch.is_tensor(hi_mod) else hi_mod
        sos_mod = design_butter_sos_scipy(4, [lo_val, hi_val], 'bandpass')
        sos_level = design_butter_sos_scipy(2, level_lp_hz, 'lowpass')
        sos_env = None
        if envelope_method == "rectify_lp":
            sos_env = design_butter_sos_scipy(2, env_lp_hz, 'lowpass')
        
    # Custom Differentiable SOS Filtering (Warning: Slow in Python loop)
    # But necessary if torchaudio.functional.sosfilt is missing.
    
    def custom_sosfilt(x, sos):
        """
        Differentiable SOS filter implementation for tensors.
        x: (..., Time)
        sos: (..., Sections, 6)
        Returns: filtered x
        """
        # We need to filter each section sequentially.
        # Direct Form II Transposed or Direct Form I?
        # Direct Form I is easiest to implement nicely in a loop for gradients.
        # coefficients: b0, b1, b2, a0, a1, a2
        
        # Normalize by a0
        # sos comes in shape (..., n_sections, 6)
        # We assume x matches batching of sos, or broadcasting.
        
        # Let's simplify: loop over sections
        n_sections = sos.shape[-2]
        y = x
        
        for s in range(n_sections):
             coeffs = sos[..., s, :] # (..., 6)
             b0, b1, b2 = coeffs[..., 0], coeffs[..., 1], coeffs[..., 2]
             a0, a1, a2 = coeffs[..., 3], coeffs[..., 4], coeffs[..., 5]
             
             # Normalize
             b0 = b0 / (a0 + 1e-8)
             b1 = b1 / (a0 + 1e-8)
             b2 = b2 / (a0 + 1e-8)
             a1 = a1 / (a0 + 1e-8)
             a2 = a2 / (a0 + 1e-8)
             
             # Time loop (very slow in python, but valid grad)
             # Or use torchaudio.functional.lfilter if available?
             # Check for lfilter
             if hasattr(TAF, 'lfilter'):
                 # lfilter takes (a, b)
                 # a = [1, a1, a2]
                 # b = [b0, b1, b2]
                 # TAF.lfilter expects a_coeffs, b_coeffs.
                 # Shapes? TAF.lfilter(waveform, a_coeffs, b_coeffs)
                 # waveform: (..., time)
                 # coeffs: (..., order+1) or 1D
                 
                 # Construct coeffs tensor
                 # We need to stack them.
                 # Note: TAF.lfilter might accept batch params?
                 # Docs usually say coeffs are 1D.
                 # If params are batched, we might need a loop or TAF support.
                 
                 # Let's assume we can loop over sections and use TAF.lfilter if params are constant across batch?
                 # If params vary per batch, TAF.lfilter might not support it (check version).
                 # For tuning, params are usually global (scalar).
                 
                 b_poly = torch.stack([b0, b1, b2], dim=-1).to(y.dtype)
                 a_poly = torch.stack([torch.ones_like(a0), a1, a2], dim=-1).to(y.dtype)
                 
                 # If b_poly has batch dims but y is (B, T), TAF.lfilter might fail if it expects 1D coeffs.
                 # For training, we usually have 1 set of params for the whole batch.
                 if b_poly.ndim > 1:
                     # Remove dimensions 1 if they are singleton?
                     b_poly = b_poly.squeeze()
                     a_poly = a_poly.squeeze()
                 
                 y = TAF.lfilter(y, a_poly, b_poly)
             else:
                 # Naive recurrence implementation
                 # y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
                 # This is O(T) python ops. Very slow.
                 # But functional for "tuning" short signals.
                 
                 # Use inputs x_curr = y
                 out = []
                 x_n1 = torch.zeros_like(y[..., 0])
                 x_n2 = torch.zeros_like(y[..., 0])
                 y_n1 = torch.zeros_like(y[..., 0])
                 y_n2 = torch.zeros_like(y[..., 0])
                 
                 # Loop time
                 T = y.shape[-1]
                 for t in range(T):
                     x_val = y[..., t]
                     output = b0*x_val + b1*x_n1 + b2*x_n2 - a1*y_n1 - a2*y_n2
                     out.append(output)
                     
                     x_n2 = x_n1
                     x_n1 = x_val
                     y_n2 = y_n1
                     y_n1 = output
                 
                 y = torch.stack(out, dim=-1)
        
        return y

    def torch_sosfiltfilt(x, sos):
        """
        Zero-phase filtering.
        """
        # Check if we can use scipy (non-differentiable)
        if not sos.requires_grad and not x.requires_grad:
             x_np = x.detach().cpu().numpy()
             sos_np = sos.detach().cpu().numpy()
             try:
                 y_np = scipy.signal.sosfiltfilt(sos_np, x_np, axis=-1)
                 return torch.from_numpy(y_np.copy()).to(device=x.device, dtype=x.dtype)
             except Exception as e:
                 # Fallback to custom if SciPy fails (e.g. singular matrix)
                 # This is slower but robust.
                 pass
            
        # Differentiable Path
        # Forward
        y = custom_sosfilt(x, sos)
        # Backward
        y = torch.flip(y, dims=[-1])
        y = custom_sosfilt(y, sos)
        y = torch.flip(y, dims=[-1])
        return y

    win_samp = int(round(rough_rms_win_ms * 1e-3 * fs))
    win_samp = max(win_samp, 1)
    
    R_accum = torch.zeros_like(x)
    
    # Process bands
    for fc in centers:
        # calculate band limits
        fc_val = fc if torch.is_tensor(fc) else torch.tensor(fc, device=device)
        bw = erb_bw_scale * erb_bw_hz(fc_val)
        lo_c = fc_val - 0.5 * bw
        hi_c = fc_val + 0.5 * bw
        
        # Hard clamp freq - tighter for differentiable path to avoid tan() overflow
        max_freq = 0.45 * nyq if needs_grad else 0.95 * nyq
        lo_c = torch.clamp(lo_c, min=1.0)
        hi_c = torch.clamp(hi_c, max=max_freq)
        
        if hi_c <= lo_c: continue
            
        # Design band filter (SciPy for inference, diff for training)
        if needs_grad:
            sos_bp = diff_butter_bandpass_sos(lo_c, hi_c, fs, order=bandpass_order).to(device)
        else:
            sos_bp = design_butter_sos_scipy(bandpass_order, [lo_c.item(), hi_c.item()], 'bandpass')
        
        # 1. Bandpass
        y = torch_sosfiltfilt(x, sos_bp)
        if torch.isnan(y).any():
            print(f"FIRST NaN at fc={fc_val.item():.1f}Hz, lo_c={lo_c.item():.1f}, hi_c={hi_c.item():.1f}")
            # Skip this band
            continue
        
        # 2. Envelope
        if envelope_method == "hilbert":
            env = hilbert_envelope(y)
        elif envelope_method == "rectify_lp":
            env = torch.abs(y)
            env = torch_sosfiltfilt(env, sos_env)
        else:
            raise ValueError(f"Unknown envelope method: {envelope_method}")
            
        # 3. Slow Level
        level = torch_sosfiltfilt(env, sos_level)
        
        # 4. Modulation (Roughness Driver)
        mod = torch_sosfiltfilt(env, sos_mod)
             
        # 5. Local RMS of mod
        rough = moving_rms(mod, win_samp)
        
        # 6. Weighting
        weight = torch.pow(torch.clamp(level, min=0.0), level_compression)
        
        R_accum = R_accum + weight * rough
        
    R_accum = R_accum / max(len(centers), 1)
    
    # Resample / Decimate
    # Resample / Decimate
    if output_hop_s is None:
        t_out = torch.arange(R_accum.shape[-1], device=device) / fs
        return t_out, R_accum, centers
        
    hop = int(round(output_hop_s * fs))
    hop = max(hop, 1)
    
    # Average pooling for downsampling
    # Pad to ensure divisibility? Or trim as in ref? Ref trims.
    n_samples = (R_accum.shape[-1] // hop) * hop
    R_trim = R_accum[..., :n_samples]
    
    # Reshape (..., Frames, Hop)
    # We want to keep leading batch dims
    batch_dims = R_trim.shape[:-1]
    R_reshaped = R_trim.view(*batch_dims, -1, hop)
    R_out = R_reshaped.mean(dim=-1) # (..., Frames)
    
    t_out = (torch.arange(R_out.shape[-1], device=device) * hop) / fs
    
    return t_out, R_out, centers
