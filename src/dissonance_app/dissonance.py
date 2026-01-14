import torch
import math

def precompute_dissonance_matrix(freqs, alpha=0.021, beta=19.0, ignore_bins=4):
    """
    Precompute the pairwise dissonance interaction matrix for a set of frequencies.
    Returns matrix D of shape (n_freqs, n_freqs).
    D[i, j] represents the roughness of two partials at f[i], f[j] with amplitudes 1.
    ignore_bins: Number of bins to ignore around the diagonal (band width). 
                 interactions where |i-j| <= ignore_bins will be set to 0. 
                 Helps ignore spectral leakage self-dissonance.
    """
    # Create meshgrid of frequencies
    f1 = freqs.unsqueeze(1) # Column
    f2 = freqs.unsqueeze(0) # Row
    
    # Sethares formula depends on min(f1, f2) and abs diff
    f_min = torch.min(f1, f2)
    f_diff = torch.abs(f1 - f2)
    
    # Avoid division by zero if freq is 0 (DC component)
    div = alpha * f_min + beta
    
    s = f_diff / div
    
    term1 = torch.exp(-3.5 * s)
    term2 = torch.exp(-5.75 * s)
    
    d_mat = term1 - term2
    
    # Zero out diagonal (self-dissonance is 0 in this model)
    d_mat.fill_diagonal_(0)
    
    # Mask leakage band
    if ignore_bins > 0:
        # Create indices
        n = len(freqs)
        rows = torch.arange(n).unsqueeze(1).to(freqs.device)
        cols = torch.arange(n).unsqueeze(0).to(freqs.device)
        dist = torch.abs(rows - cols)
        mask = dist <= ignore_bins
        d_mat[mask] = 0.0
    
    return d_mat

def timbral_dissonance(audio: torch.Tensor, sr: int, n_fft: int = 4096, hop_length: int = 512, alpha: float = 0.021, beta: float = 19.0, ignore_bins: int = 4):
    """
    Compute the dissonance curve and integral for a raw audio signal using dense STFT interactions.
    
    Args:
        audio: (Tensor) Raw audio samples, 1D or (1, T).
        sr: (int) Sample rate.
        n_fft: (int) FFT size. Default increased to 4096 for better freq resolution.
        hop_length: (int) STFT hop length.
        alpha: (float) Scale factor for critical bandwidth (Sethares default 0.021).
        beta: (float) Offset for critical bandwidth (Sethares default 19.0).
        ignore_bins: (int) Number of adjacent bins to ignore (diagonal filtering).
        
    Returns:
        dissonance_curve: (Tensor) 1D tensor of dissonance values over time.
        dissonance_integral: (float) Sum of dissonance values (normalized via energy).
    """
    if audio.dim() == 2:
        audio = audio.squeeze(0)
        
    device = audio.device
    
    # STFT
    window = torch.hann_window(n_fft).to(device)
    stft = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    magnitude = torch.abs(stft).transpose(0, 1) # [frames, bins]
    
    # Frequencies
    freqs = torch.fft.rfftfreq(n_fft, d=1/sr).to(device)
    
    # Precompute interaction matrix
    D = precompute_dissonance_matrix(freqs, alpha=alpha, beta=beta, ignore_bins=ignore_bins) # [bins, bins]
    
    # Compute Frame Dissonance
    # Dissonance[t] = sum_{i,j} mag[t,i] * mag[t,j] * D[i,j]
    #               = mag[t] @ D @ mag[t].T
    # Efficient calculation: (mag @ D) * mag -> sum over last dim
    
    # mag: [frames, bins]
    # D: [bins, bins]
    # mag @ D -> [frames, bins]
    interacted = torch.matmul(magnitude, D) 
    
    # Element-wise multiply with mag and sum
    dissonance_curve = torch.sum(interacted * magnitude, dim=1) # [frames]
    
    # Normalization
    # User requirement: scaling audio by K should not change result.
    # D scales by K^2.
    # We normalized by Energy = sum(mag^2). Energy scales by K^2.
    # Ratio is invariant.
    
    energy_curve = torch.sum(magnitude ** 2, dim=1)
    
    # Avoid div by zero
    safe_energy = torch.where(energy_curve > 1e-9, energy_curve, torch.tensor(1.0, device=device))
    normalized_curve = dissonance_curve / safe_energy
    
    # Zero out silent frames
    normalized_curve = torch.where(energy_curve > 1e-9, normalized_curve, torch.tensor(0.0, device=device))
    
    dissonance_integral = torch.sum(normalized_curve).item()
    
    return normalized_curve, dissonance_integral
