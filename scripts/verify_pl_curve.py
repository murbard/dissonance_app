import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dissonance_app.dissonance import timbral_dissonance

def sethares_pair_dissonance(f1, f2, alpha=0.021, beta=19.0):
    """Theoretical dissonance between two partials with amp=1."""
    f_min = min(f1, f2)
    f_diff = abs(f1 - f2)
    s = f_diff / (alpha * f_min + beta)
    return np.exp(-3.5 * s) - np.exp(-5.75 * s)

def run_sweep():
    sr = 22050
    base_freq = 440.0
    # Sweep from Unison (ratio 1.0) to Octave (ratio 2.0)
    ratios = np.linspace(1.0, 2.3, 100)
    
    computed_dissonance = []
    theoretical_dissonance = []
    
    print(f"Sweeping intervals from unison to >octave ({len(ratios)} steps)...")
    
    for r in ratios:
        f2 = base_freq * r
        
        # Generator
        duration = 0.5
        t = torch.linspace(0, duration, int(sr * duration))
        
        # Construct signal: base + interval
        # Amplitudes = 1.0
        sig = torch.sin(2 * 3.14159 * base_freq * t) + torch.sin(2 * 3.14159 * f2 * t)
        
        # Compute
        curve, integral = timbral_dissonance(sig, sr)
        computed_dissonance.append(integral)
        
        # Theoretical (Sethares)
        # 2 sines => 3 terms: D(f1,f1), D(f2,f2), D(f1,f2).
        # Sethares usually defines D(f1,f1)=0. 
        # Our model defines diagonal=0 so self-dissonance is 0 (ignoring leakage).
        # So we expect proportional to D(f1, f2).
        # Note: Logic sums D_ij, so we get D(f1,f2) + D(f2,f1) = 2 * value.
        d_theory = 2 * sethares_pair_dissonance(base_freq, f2)
        theoretical_dissonance.append(d_theory)
        
    # Correlation
    comp = np.array(computed_dissonance)
    theo = np.array(theoretical_dissonance)
    
    # Normalize for plotting
    comp_norm = comp / comp.max()
    theo_norm = theo / theo.max()
    
    correlation = np.corrcoef(comp, theo)[0, 1]
    print(f"Correlation between Computed and Theoretical: {correlation:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ratios, comp_norm, label='Computed (STFT Dense)', linewidth=2)
    plt.plot(ratios, theo_norm, label='Theoretical (Sethares)', linestyle='--', linewidth=2)
    plt.title(f'Plomp-Levelt Curve Verification (Corr: {correlation:.4f})')
    plt.xlabel('Frequency Ratio (f2/f1)')
    plt.ylabel('Normalized Dissonance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('plomp_levelt_verification.png')
    print("Saved plot to plomp_levelt_verification.png")
    
    if correlation < 0.9:
        print("WARNING: Correlation is low. Check implementation.")
    else:
        print("SUCCESS: Shape matches theory.")

if __name__ == "__main__":
    run_sweep()
