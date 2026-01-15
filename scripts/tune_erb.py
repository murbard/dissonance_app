import torch
import numpy as np
import scipy.optimize
from dissonance_app.roughness_erb import roughness_curve_erb

def theoretical_curve(ratios, base_freq):
    vals = []
    for r in ratios:
        f2 = base_freq * r
        f_diff = abs(f2 - base_freq)
        f_min = base_freq
        # Sethares params
        s = f_diff / (0.021 * f_min + 19.0)
        d = np.exp(-3.5 * s) - np.exp(-5.75 * s)
        vals.append(d)
    return np.array(vals)

def objective(params):
    # Unpack params
    # params: [erb_bw_scale, mod_lo, mod_hi, level_compression]
    erb_bw_scale, mod_lo, mod_hi, level_compression = params
    
    # Constraints/Bounds checks (soft penalty)
    if erb_bw_scale < 0.1 or erb_bw_scale > 3.0: return 2.0
    if mod_lo < 1.0 or mod_lo > 100.0: return 2.0
    if mod_hi < mod_lo + 10.0 or mod_hi > 500.0: return 2.0
    if level_compression < 0.1 or level_compression > 2.0: return 2.0
    
    sr = 22050 # Lower SR for speed during tuning
    base_freq = 440.0
    ratios = np.linspace(1.0, 2.3, 20) # Coarse grid
    
    erb_vals = []
    
    # Generate signals
    for r in ratios:
        f2 = base_freq * r
        # 0.5s duration
        t = torch.linspace(0, 0.5, int(sr * 0.5))
        sig = torch.sin(2 * torch.pi * base_freq * t) + torch.sin(2 * torch.pi * f2 * t)
        
        try:
            _, R_curve, _ = roughness_curve_erb(
                sig, sr,
                erb_bw_scale=erb_bw_scale,
                mod_band_hz=(mod_lo, mod_hi),
                level_compression=level_compression,
                n_bands=32, # Reduced bands for speed optimization
                output_hop_s=None
            )
            # Take mean
            if R_curve.numel() > 0:
                erb_vals.append(R_curve.mean().item())
            else:
                erb_vals.append(0.0)
        except Exception as e:
            # print(f"Error with params {params}: {e}")
            return 2.0
            
    erb_vals = np.array(erb_vals)
    theo_vals = theoretical_curve(ratios, base_freq)
    
    # Correlation
    if np.std(erb_vals) < 1e-9: return 2.0 # Flat line
    
    corr = np.corrcoef(erb_vals, theo_vals)[0, 1]
    
    # We want to MAXIMIZE correlation, so MINIMIZE (1 - corr)
    # If correlation is negative, result is > 1.
    return 1.0 - corr

def run_tuning():
    print("Starting optimization...")
    # Initial guess: default-ish
    # [scale, lo, hi, comp]
    x0 = [1.019, 20.0, 200.0, 0.3]
    
    # Bounds for Powell (handled via penalty in objective usually, 
    # but scipy.optimize can take bounds for some methods like L-BFGS-B or TNC, 
    # Powell doesn't strictly support bounds but we have penalty)
    # Let's use 'Nelder-Mead' or 'Powell' with penalty.
    
    res = scipy.optimize.minimize(objective, x0, method='Nelder-Mead', tol=1e-3, options={'disp': True, 'maxiter': 50})
    
    print("\nOptimization Result:")
    print(f"Success: {res.success}")
    print(f"Best Correlation: {1.0 - res.fun:.4f}")
    print(f"Params: bw_scale={res.x[0]:.4f}, mod_lo={res.x[1]:.4f}, mod_hi={res.x[2]:.4f}, comp={res.x[3]:.4f}")

if __name__ == "__main__":
    run_tuning()
