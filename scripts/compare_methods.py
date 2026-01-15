import torch
import numpy as np
import matplotlib.pyplot as plt
from dissonance_app.roughness_erb import roughness_curve_erb

def generate_sine(freq, sr, duration, device='cpu'):
    t = torch.linspace(0, duration, int(sr * duration), device=device)
    return torch.sin(2 * torch.pi * freq * t)

def compare_methods():
    sr = 44100
    base_freq = 440.0
    # User requested 200 points
    ratios = np.linspace(1.0, 2.3, 200)
    
    erb_vals = []
    theo_vals = []
    
    print(f"Sweeping intervals (SR={sr}, 200 points)...")
    
    # Force CPU
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    for r in ratios:
        f2 = base_freq * r
        # 1.0 second duration
        sig = generate_sine(base_freq, sr, 1.0, device=device) + generate_sine(f2, sr, 1.0, device=device)
        
        # Method: ERB (Time Domain)
        _, R_curve, _ = roughness_curve_erb(sig, sr)
        erb_scalar = torch.mean(R_curve).item()
        erb_vals.append(erb_scalar)
        
        # Method: Theoretical (Sethares)
        f_diff = abs(f2 - base_freq)
        f_min = base_freq
        s = f_diff / (0.021 * f_min + 19.0)
        d = np.exp(-3.5 * s) - np.exp(-5.75 * s)
        theo_vals.append(d)
        
    # Normalize
    erb_vals = np.array(erb_vals)
    theo_vals = np.array(theo_vals)
    
    erb_norm = (erb_vals - erb_vals.min()) / (erb_vals.max() - erb_vals.min() + 1e-9)
    theo_norm = (theo_vals - theo_vals.min()) / (theo_vals.max() - theo_vals.min() + 1e-9)
    
    corr_erb_theo = np.corrcoef(erb_norm, theo_norm)[0, 1]
    
    print(f"Correlation ERB vs Theo: {corr_erb_theo:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(ratios, erb_norm, label=f'Roughness ERB (Corr={corr_erb_theo:.2f})', linewidth=2)
    plt.plot(ratios, theo_norm, label='Theoretical Sethares', linestyle='--', color='black', alpha=0.7)
    
    plt.title(f'ERB Roughness vs Theoretical Plomp-Levelt')
    plt.xlabel('Frequency Ratio')
    plt.ylabel('Normalized Roughness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_file = 'erb_vs_theory_200pt.png'
    plt.savefig(out_file)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    compare_methods()
