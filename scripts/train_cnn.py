"""
Deep CNN for roughness estimation.
Replaces the ERB-based model with a 1D CNN that has more capacity.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os


def sample_f0_piano(n_samples):
    """Sample f0 from piano range using Beta(7,7) distribution."""
    b = np.random.beta(7, 7, n_samples)
    midi_note = 20 + 80 * b
    f0 = 440 * (2 ** ((midi_note - 69) / 12))
    return f0


def sethares_pairwise(f1, f2, a1, a2, alpha=0.021, beta=19.0):
    """Sethares pairwise dissonance between two partials."""
    f_min = min(f1, f2)
    f_diff = abs(f1 - f2)
    s = f_diff / (alpha * f_min + beta)
    d = a1 * a2 * (np.exp(-3.5 * s) - np.exp(-5.75 * s))
    return max(0, d)


def get_theoretical_dissonance(f0, ratio, n_harmonics=12, amp_power=1.0, inharmonicity=None):
    """Sethares dissonance for two harmonic tones."""
    f1_base = f0
    f2_base = f0 * ratio
    
    freqs1, freqs2 = [], []
    for n in range(1, n_harmonics + 1):
        if inharmonicity is not None:
            shift = inharmonicity[n - 1]
            freqs1.append(f1_base * n * np.exp(shift))
            freqs2.append(f2_base * n * np.exp(shift))
        else:
            freqs1.append(f1_base * n)
            freqs2.append(f2_base * n)
    
    amps1 = [1.0 / (n ** amp_power) for n in range(1, n_harmonics + 1)]
    amps2 = [1.0 / (n ** amp_power) for n in range(1, n_harmonics + 1)]
    
    total_d = 0.0
    for fi, ai in zip(freqs1, amps1):
        for fj, aj in zip(freqs2, amps2):
            total_d += sethares_pairwise(fi, fj, ai, aj)
    return total_d


def generate_harmonic_tone(f0, t, n_harmonics=12, amp_power=1.0, inharmonicity=None):
    """Generate a harmonic tone with configurable amplitude falloff and inharmonicity."""
    sig = torch.zeros_like(t)
    for n in range(1, n_harmonics + 1):
        amp = 1.0 / (n ** amp_power)
        if inharmonicity is not None:
            freq = f0 * n * torch.exp(inharmonicity[n - 1])
        else:
            freq = f0 * n
        sig = sig + amp * torch.sin(2 * torch.pi * freq * t)
    return sig


class RoughnessCNN(nn.Module):
    """
    1D CNN for roughness estimation.
    Uses strided convolutions to progressively downsample, then global pooling.
    """
    def __init__(self, base_channels=32):
        super().__init__()
        
        # Initial projection
        self.conv1 = nn.Conv1d(1, base_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(base_channels)
        
        # Downsampling blocks
        self.block1 = self._make_block(base_channels, base_channels * 2, stride=4)
        self.block2 = self._make_block(base_channels * 2, base_channels * 4, stride=4)
        self.block3 = self._make_block(base_channels * 4, base_channels * 8, stride=4)
        self.block4 = self._make_block(base_channels * 8, base_channels * 8, stride=4)
        
        # Global pooling + output
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def _make_block(self, in_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, x):
        # x: (batch, samples)
        x = x.unsqueeze(1)  # (batch, 1, samples)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        return self.head(x).squeeze(-1)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    sr = 22050
    model = RoughnessCNN(base_channels=32).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Learnable parameters: {n_params}")
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    
    # Load checkpoint if exists
    checkpoint_path = 'best_cnn_roughness.pt'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if 'model_state_dict' in checkpoint:
            result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if result.missing_keys or result.unexpected_keys:
                print(f"Partial load - missing: {len(result.missing_keys)}, unexpected: {len(result.unexpected_keys)}")
                best_val_loss = float('inf')
            else:
                best_val_loss = checkpoint.get('val_loss', float('inf'))
                print(f"Loaded checkpoint, best_val_loss: {best_val_loss:.5f}")
        else:
            best_val_loss = float('inf')
    else:
        best_val_loss = float('inf')
        print("No checkpoint found, starting fresh.")
    
    # Validation frequencies
    val_f0s = [102, 126, 147, 167, 186, 207, 227, 250, 274, 301, 331, 367, 410, 465, 541, 670]
    val_ratios = np.linspace(1.0, 2.25, 100)
    
    batch_size = 64
    n_steps = 20000
    mse_criterion = nn.MSELoss()
    n_harmonics = 12
    
    print("Starting CNN training...")
    
    for step in range(n_steps):
        model.train()
        optimizer.zero_grad()
        
        # Sample f0 from piano range
        f0s = sample_f0_piano(batch_size)
        
        # Sample ratios (up to 14 semitones)
        ratios = 2 ** (14 * np.random.uniform(size=batch_size) / 12)
        
        # Sample amplitude profiles: 1.0 (sawtooth) or 2.0 (softer)
        amp_powers = np.random.choice([1.0, 2.0], size=batch_size)
        
        # Sample inharmonicity: ±3% log-frequency shift per harmonic
        inharmonicities = np.random.uniform(-0.03, 0.03, size=(batch_size, n_harmonics))
        
        # Compute targets
        targets = torch.tensor([
            get_theoretical_dissonance(f0, r, n_harmonics=n_harmonics, amp_power=ap, inharmonicity=inh)
            for f0, r, ap, inh in zip(f0s, ratios, amp_powers, inharmonicities)
        ], dtype=torch.float32).to(device)
        
        # Generate signals
        t = torch.linspace(0, 0.5, int(sr * 0.5), device=device)
        signals = []
        for f0, r, ap, inh in zip(f0s, ratios, amp_powers, inharmonicities):
            f1 = f0 * r
            inh_t = torch.tensor(inh, dtype=torch.float32, device=device)
            sig = generate_harmonic_tone(f0, t, n_harmonics=n_harmonics, amp_power=ap, inharmonicity=inh_t) + \
                  generate_harmonic_tone(f1, t, n_harmonics=n_harmonics, amp_power=ap, inharmonicity=inh_t)
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
            
            scheduler.step(val_loss)
            print(f"Step {step}: TrainLoss {loss.item():.5f}, ValLoss {val_loss:.5f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  >>> NEW BEST!")
                save_plot(model, val_f0s, val_ratios, sr, device, val_loss, step)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'step': step
                }, checkpoint_path)
    
    print(f"\nDone. Best ValLoss: {best_val_loss:.5f}")


def compute_val_loss(model, f0s, ratios, sr, device, criterion):
    total_loss = 0.0
    t = torch.linspace(0, 0.5, int(sr * 0.5), device=device)
    
    for f0 in f0s:
        targets = torch.tensor([get_theoretical_dissonance(f0, r) for r in ratios], dtype=torch.float32).to(device)
        signals = torch.stack([
            generate_harmonic_tone(f0, t) + generate_harmonic_tone(f0 * r, t) for r in ratios
        ])
        preds = model(signals)
        
        p_norm = (preds - preds.min()) / (preds.max() - preds.min() + 1e-8)
        t_norm = (targets - targets.min()) / (targets.max() - targets.min() + 1e-8)
        total_loss += criterion(p_norm, t_norm).item()
    
    return total_loss / len(f0s)


def save_plot(model, f0s, ratios, sr, device, loss, step):
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    t = torch.linspace(0, 0.5, int(sr * 0.5), device=device)
    
    for i, f0 in enumerate(f0s):
        ax = axes[i]
        
        theory = np.array([get_theoretical_dissonance(f0, r) for r in ratios])
        theory_norm = (theory - theory.min()) / (theory.max() - theory.min() + 1e-8)
        
        with torch.no_grad():
            signals = torch.stack([
                generate_harmonic_tone(f0, t) + generate_harmonic_tone(f0 * r, t) for r in ratios
            ])
            preds = model(signals).cpu().numpy()
        preds_norm = (preds - preds.min()) / (preds.max() - preds.min() + 1e-8)
        
        ax.plot(ratios, theory_norm, 'k-', lw=2, label='Theoretical')
        ax.plot(ratios, preds_norm, 'r--', lw=2, label='CNN Model')
        ax.set_xlabel('Frequency Ratio')
        ax.set_ylabel('Normalized Dissonance')
        ax.set_title(f'f0 = {f0:.0f} Hz')
        ax.legend()
        ax.grid(alpha=0.3)
    
    fig.suptitle(f'CNN Roughness Model (Step {step}, Val Loss: {loss:.5f})', fontsize=14)
    plt.tight_layout()
    plt.savefig('best_cnn_fit.png', dpi=100)
    plt.close()


if __name__ == "__main__":
    train()
