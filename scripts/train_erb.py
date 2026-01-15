"""
Sine basis filter training for ERB roughness optimization.
Multi-f0 version: trains across piano range using Beta(7,7) distribution.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import os


def sample_f0_piano(n_samples):
    """
    Sample f0 from piano range using Beta(7,7) distribution.
    f0 = 440 * 2^((20 + 80*b - 69) / 12) where b ~ Beta(7,7)
    """
    b = np.random.beta(7, 7, n_samples)
    midi_note = 20 + 80 * b
    f0 = 440 * (2 ** ((midi_note - 69) / 12))
    return f0


def get_theoretical_dissonance(f0, ratio):
    """Plomp-Levelt curve for a specific f0 and ratio."""
    f2 = f0 * ratio
    f_diff = abs(f2 - f0)
    f_min = min(f0, f2)
    s = f_diff / (0.021 * f_min + 19.0)
    d = np.exp(-3.5 * s) - np.exp(-5.75 * s)
    return d


class SineBasisFilter(nn.Module):
    """Filter parameterized by sine basis: h[n] = Σₖ bₖ · sin(πk·n/N)"""
    def __init__(self, filter_length, n_coeffs, init_from_scipy=None):
        super().__init__()
        self.filter_length = filter_length
        self.n_coeffs = n_coeffs
        
        t = torch.linspace(0, 1, filter_length)
        k = torch.arange(1, n_coeffs + 1).float()
        basis = torch.sin(np.pi * k.unsqueeze(0) * t.unsqueeze(1))
        self.register_buffer('basis', basis)
        
        if init_from_scipy is not None:
            init_coeffs = self._project_to_sine_basis(init_from_scipy)
        else:
            init_coeffs = torch.randn(n_coeffs) * 0.01
        
        self.coeffs = nn.Parameter(init_coeffs)
    
    def _project_to_sine_basis(self, filter_taps):
        if len(filter_taps) != self.filter_length:
            filter_taps = np.interp(
                np.linspace(0, 1, self.filter_length),
                np.linspace(0, 1, len(filter_taps)),
                filter_taps
            )
        basis_np = self.basis.numpy()
        coeffs, _, _, _ = np.linalg.lstsq(basis_np, filter_taps, rcond=None)
        return torch.from_numpy(coeffs).float()
    
    def get_taps(self):
        return self.basis @ self.coeffs
    
    def forward(self, x):
        """Apply filter via FFT convolution (much faster for long filters)."""
        taps = self.get_taps()
        
        # FFT convolution
        n_fft = x.shape[-1] + self.filter_length - 1
        # Round up to power of 2 for efficiency
        n_fft = 2 ** int(np.ceil(np.log2(n_fft)))
        
        X = torch.fft.rfft(x, n=n_fft)
        H = torch.fft.rfft(taps, n=n_fft)
        Y = X * H
        y = torch.fft.irfft(Y, n=n_fft)
        
        # Trim to original length (centered)
        start = self.filter_length // 2
        y = y[..., start:start + x.shape[-1]]
        
        return y


class ERBSineBasisModel(nn.Module):
    """ERB roughness model using sine basis filters."""
    def __init__(self, fs=22050, n_bands=16, filter_length_ms=50, n_coeffs=16):
        super().__init__()
        self.fs = fs
        self.n_bands = n_bands
        self.filter_length = int(filter_length_ms * fs / 1000)
        
        # Modulation bandpass (20-200 Hz)
        mod_fir = scipy.signal.firwin(self.filter_length, [20, 200], pass_zero=False, fs=fs)
        self.mod_filter = SineBasisFilter(self.filter_length, n_coeffs, init_from_scipy=mod_fir)
        
        # Envelope lowpass (100 Hz)
        env_fir = scipy.signal.firwin(self.filter_length, 100, fs=fs)
        self.env_filter = SineBasisFilter(self.filter_length, n_coeffs, init_from_scipy=env_fir)
        
        # Level lowpass (10 Hz)
        level_length = int(200 * fs / 1000)
        level_fir = scipy.signal.firwin(level_length, 10, fs=fs)
        self.level_filter = SineBasisFilter(level_length, n_coeffs, init_from_scipy=level_fir)
        
        # Per-band bandpass - NOW LEARNABLE
        fmin, fmax = 20.0, min(10000.0, fs/2 * 0.9)
        erb_lo = 21.4 * np.log10(0.00437 * fmin + 1)
        erb_hi = 21.4 * np.log10(0.00437 * fmax + 1)
        erb_centers = np.linspace(erb_lo, erb_hi, n_bands)
        hz_centers = (10 ** (erb_centers / 21.4) - 1) / 0.00437
        
        self.band_filters = nn.ModuleList()
        for fc in hz_centers:
            bw = 24.7 + 0.108 * fc
            lo = max(1.0, fc - bw/2)
            hi = min(fs/2 * 0.95, fc + bw/2)
            if hi > lo:
                band_fir = scipy.signal.firwin(self.filter_length, [lo, hi], pass_zero=False, fs=fs)
            else:
                band_fir = np.zeros(self.filter_length)
                band_fir[self.filter_length // 2] = 1.0
            self.band_filters.append(SineBasisFilter(self.filter_length, n_coeffs, init_from_scipy=band_fir))
        
        self.level_compression = nn.Parameter(torch.tensor(0.3))
        self.rms_win = int(10e-3 * fs)
    
    def _apply_band_filter(self, x, band_idx):
        return self.band_filters[band_idx](x)
    
    def forward(self, x):
        batch_size, n_samples = x.shape
        R_accum = torch.zeros_like(x)
        
        for b in range(self.n_bands):
            y = self._apply_band_filter(x, b)
            env = torch.abs(y)
            env = self.env_filter(env)
            level = self.level_filter(env)
            mod = self.mod_filter(env)
            
            mod_sq = mod * mod
            rms_sq = F.avg_pool1d(mod_sq.unsqueeze(1), self.rms_win, stride=1, padding=self.rms_win//2).squeeze(1)
            if rms_sq.shape[-1] > n_samples:
                rms_sq = rms_sq[..., :n_samples]
            rough = torch.sqrt(rms_sq + 1e-10)
            
            comp = torch.clamp(self.level_compression, 0.1, 1.0)
            weight = torch.pow(torch.clamp(level, min=1e-10), comp)
            R_accum = R_accum + weight * rough
        
        return R_accum.mean(dim=-1)


def train():
    device = torch.device('cpu')
    print(f"Training on {device}")
    
    sr = 22050
    model = ERBSineBasisModel(fs=sr, n_bands=32, filter_length_ms=100, n_coeffs=32).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Learnable parameters: {n_params}")
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Load checkpoint if exists
    checkpoint_path = 'best_erb_params.pt'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Loaded checkpoint from {checkpoint_path}, best_val_loss: {best_val_loss:.5f}")
    else:
        best_val_loss = float('inf')
        print("No checkpoint found, starting fresh.")
    
    # 9 f0 values for validation (A1 to A6)
    val_f0s = [120, 155, 188, 223, 262, 307, 363, 441, 572]
    val_ratios = np.linspace(1.0, 2.25, 100)
    
    batch_size = 256
    n_steps = 10000
    mse_criterion = nn.MSELoss()
    
    print("Starting training with varied f0...")
    
    for step in range(n_steps):
        model.train()
        optimizer.zero_grad()
        
        # Sample f0 from piano range
        f0s = sample_f0_piano(batch_size)
        
        # Sample ratios (up to 14 semitones)
        ratios = 2 ** (14 * np.random.uniform(size=batch_size) / 12)
        
        # Compute targets
        targets = torch.tensor([
            get_theoretical_dissonance(f0, r) for f0, r in zip(f0s, ratios)
        ], dtype=torch.float32).to(device)
        
        # Generate signals
        t = torch.linspace(0, 0.5, int(sr * 0.5), device=device)
        signals = []
        for f0, r in zip(f0s, ratios):
            f1 = f0 * r
            sig = torch.sin(2 * torch.pi * f0 * t) + torch.sin(2 * torch.pi * f1 * t)
            signals.append(sig)
        signals = torch.stack(signals)
        
        preds = model(signals)
        
        # Normalize
        p_norm = (preds - preds.min()) / (preds.max() - preds.min() + 1e-8)
        t_norm = (targets - targets.min()) / (targets.max() - targets.min() + 1e-8)
        
        loss = mse_criterion(p_norm, t_norm)
        
        if torch.isnan(loss):
            print(f"Step {step}: NaN loss")
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = compute_val_loss(model, val_f0s, val_ratios, sr, device, mse_criterion)
            
            print(f"Step {step}: TrainLoss {loss.item():.5f}, ValLoss {val_loss:.5f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  >>> NEW BEST!")
                save_plot(model, val_f0s, val_ratios, sr, device, val_loss, step)
    
    print(f"\nDone. Best ValLoss: {best_val_loss:.5f}")


def compute_val_loss(model, f0s, ratios, sr, device, criterion):
    total_loss = 0.0
    t = torch.linspace(0, 0.5, int(sr * 0.5), device=device)
    
    for f0 in f0s:
        targets = torch.tensor([get_theoretical_dissonance(f0, r) for r in ratios], dtype=torch.float32).to(device)
        signals = torch.stack([
            torch.sin(2 * torch.pi * f0 * t) + torch.sin(2 * torch.pi * f0 * r * t) for r in ratios
        ])
        preds = model(signals)
        
        p_norm = (preds - preds.min()) / (preds.max() - preds.min() + 1e-8)
        t_norm = (targets - targets.min()) / (targets.max() - targets.min() + 1e-8)
        total_loss += criterion(p_norm, t_norm).item()
    
    return total_loss / len(f0s)


def save_plot(model, f0s, ratios, sr, device, loss, step):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    t = torch.linspace(0, 0.5, int(sr * 0.5), device=device)
    
    for i, f0 in enumerate(f0s):
        ax = axes[i]
        
        theory = np.array([get_theoretical_dissonance(f0, r) for r in ratios])
        theory_norm = (theory - theory.min()) / (theory.max() - theory.min() + 1e-8)
        
        with torch.no_grad():
            signals = torch.stack([
                torch.sin(2 * torch.pi * f0 * t) + torch.sin(2 * torch.pi * f0 * r * t) for r in ratios
            ])
            preds = model(signals).cpu().numpy()
        preds_norm = (preds - preds.min()) / (preds.max() - preds.min() + 1e-8)
        
        ax.plot(ratios, theory_norm, 'k-', lw=2, label='Theoretical')
        ax.plot(ratios, preds_norm, 'r--', lw=2, label='ERB Model')
        ax.set_xlabel('Frequency Ratio')
        ax.set_ylabel('Normalized Dissonance')
        ax.set_title(f'f0 = {f0:.0f} Hz')
        ax.legend()
        ax.grid(alpha=0.3)
    
    fig.suptitle(f'Multi-F0 Validation (Step {step}, Val Loss: {loss:.5f})', fontsize=14)
    plt.tight_layout()
    plt.savefig('best_erb_fit.png', dpi=100)
    plt.close()


if __name__ == "__main__":
    train()
