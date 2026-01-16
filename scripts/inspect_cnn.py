"""
Visual inspection of CNN roughness model vs Sethares ground truth.
Plots 16 random examples comparing predicted vs theoretical dissonance curves.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from synth_generator import (
    sample_reference_pitch, sample_polynomial_timbre, 
    sample_inharmonicity, SyntheticSource, MultiSourceSample
)


class RoughnessCNN(nn.Module):
    """1D CNN for roughness estimation."""
    def __init__(self, base_channels=32):
        super().__init__()
        
        self.conv1 = nn.Conv1d(1, base_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(base_channels)
        
        self.block1 = self._make_block(base_channels, base_channels * 2, stride=4)
        self.block2 = self._make_block(base_channels * 2, base_channels * 4, stride=4)
        self.block3 = self._make_block(base_channels * 4, base_channels * 8, stride=4)
        self.block4 = self._make_block(base_channels * 8, base_channels * 8, stride=4)
        
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
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.head(x).squeeze(-1)


def generate_source_pair(ref_f0, n_harmonics=16):
    """Generate a pair of sources with same reference but independent timbres."""
    sources = []
    for _ in range(2):
        timbre = sample_polynomial_timbre(n_harmonics)
        inharm = sample_inharmonicity(n_harmonics, stdev_cents=10.0)
        amp = np.exp(np.random.uniform(np.log(0.5), np.log(1.0)))
        sources.append({
            'timbre': timbre,
            'inharm': inharm,
            'amp': amp
        })
    return sources


def compute_sethares_for_pair(source1, source2, f0_1, f0_2, alpha=0.021, beta=19.0):
    """Compute Sethares dissonance between two sources at given fundamentals."""
    # Get partials for source 1
    freqs1 = [f0_1 * n * np.exp(source1['inharm'][n-1]) for n in range(1, len(source1['timbre']) + 1)]
    amps1 = [source1['amp'] * source1['timbre'][n-1] for n in range(1, len(source1['timbre']) + 1)]
    
    # Get partials for source 2
    freqs2 = [f0_2 * n * np.exp(source2['inharm'][n-1]) for n in range(1, len(source2['timbre']) + 1)]
    amps2 = [source2['amp'] * source2['timbre'][n-1] for n in range(1, len(source2['timbre']) + 1)]
    
    # Compute pairwise dissonance
    total_d = 0.0
    for f1, a1 in zip(freqs1, amps1):
        for f2, a2 in zip(freqs2, amps2):
            f_min = min(f1, f2)
            f_diff = abs(f1 - f2)
            s = f_diff / (alpha * f_min + beta)
            d = a1 * a2 * (np.exp(-3.5 * s) - np.exp(-5.75 * s))
            total_d += max(0, d)
    
    return total_d


def generate_signal_for_sources(source1, source2, f0_1, f0_2, t, device):
    """Generate combined audio signal for two sources."""
    sig = torch.zeros_like(t)
    
    # Source 1
    for n in range(1, len(source1['timbre']) + 1):
        amp = source1['amp'] * source1['timbre'][n-1]
        freq = f0_1 * n * np.exp(source1['inharm'][n-1])
        sig = sig + amp * torch.sin(2 * torch.pi * freq * t)
    
    # Source 2
    for n in range(1, len(source2['timbre']) + 1):
        amp = source2['amp'] * source2['timbre'][n-1]
        freq = f0_2 * n * np.exp(source2['inharm'][n-1])
        sig = sig + amp * torch.sin(2 * torch.pi * freq * t)
    
    return sig


def evaluate_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on {device}")
    
    # Load model
    model = RoughnessCNN(base_channels=32).to(device)
    checkpoint_path = 'best_cnn_multisource.pt'
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found!")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint with val_loss: {checkpoint.get('val_loss', 'N/A')}")
    
    sr = 22050
    duration = 0.5
    t = torch.linspace(0, duration, int(sr * duration), device=device)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    np.random.seed(None)  # Fresh random samples each run
    
    for idx in range(16):
        ax = axes[idx]
        
        # Sample reference pitch
        ref_f0 = sample_reference_pitch()
        
        # Generate source pair
        source1, source2 = generate_source_pair(ref_f0)
        
        # Random segment: pick center between -0.35 and +0.35, then ±0.15 around it
        segment_center = np.random.uniform(-0.35, 0.35)
        segment_half = 0.15
        segment_start = segment_center - segment_half
        segment_end = segment_center + segment_half
        
        octave_shifts = np.linspace(segment_start, segment_end, 100)
        ratios = 2 ** octave_shifts
        
        # Compute theoretical Sethares curve
        theory = []
        for ratio in ratios:
            f0_2 = ref_f0 * ratio
            d = compute_sethares_for_pair(source1, source2, ref_f0, f0_2)
            theory.append(d)
        theory = np.array(theory)
        
        # Compute CNN predictions
        cnn_preds = []
        with torch.no_grad():
            for ratio in ratios:
                f0_2 = ref_f0 * ratio
                sig = generate_signal_for_sources(source1, source2, ref_f0, f0_2, t, device)
                pred = model(sig.unsqueeze(0)).item()
                cnn_preds.append(pred)
        cnn_preds = np.array(cnn_preds)
        
        # Normalize to [0, 1]
        theory_norm = (theory - theory.min()) / (theory.max() - theory.min() + 1e-8)
        cnn_norm = (cnn_preds - cnn_preds.min()) / (cnn_preds.max() - cnn_preds.min() + 1e-8)
        
        # Plot
        ax.plot(octave_shifts, theory_norm, 'k-', lw=2, label='Sethares')
        ax.plot(octave_shifts, cnn_norm, 'r--', lw=2, label='CNN')
        if segment_start <= 0 <= segment_end:
            ax.axvline(0, color='g', linestyle=':', alpha=0.5)
        ax.set_xlabel('Octave Shift')
        ax.set_ylabel('Normalized Dissonance')
        ax.set_title(f'f0={ref_f0:.0f}Hz')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    fig.suptitle('CNN vs Sethares Dissonance (16 Random Examples)', fontsize=14)
    plt.tight_layout()
    plt.savefig('cnn_vs_sethares_inspection.png', dpi=100)
    print("Saved to cnn_vs_sethares_inspection.png")
    plt.close()


if __name__ == "__main__":
    evaluate_model()
