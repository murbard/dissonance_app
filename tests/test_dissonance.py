import torch
import pytest
import math
import numpy as np
from dissonance_app.dissonance import timbral_dissonance

def generate_sine(freq, sr=22050, duration=1.0):
    t = torch.linspace(0, duration, int(sr * duration))
    return torch.sin(2 * math.pi * freq * t)

def test_scale_invariance():
    sr = 22050
    sig = generate_sine(440, sr) + 0.5 * generate_sine(450, sr)
    curve1, integral1 = timbral_dissonance(sig, sr)
    curve2, integral2 = timbral_dissonance(sig * 2.5, sr)
    assert torch.allclose(curve1, curve2, atol=1e-5)
    assert math.isclose(integral1, integral2, rel_tol=1e-5)

def test_pure_sine_relative_dissonance():
    sr = 22050
    # Consonant (Sine) vs Dissonant (Minor Second)
    sig_sine = generate_sine(440, sr)
    sig_dissonant = generate_sine(440, sr) + generate_sine(440 * 2**(1/12), sr)
    
    _, int_sine = timbral_dissonance(sig_sine, sr)
    _, int_diss = timbral_dissonance(sig_dissonant, sr)
    
    assert int_sine < int_diss, "Pure sine should be less dissonant than minor second"

def test_dissonance_interval():
    sr = 22050
    # Minor second vs Octave
    f_base = 440
    sig_bad = generate_sine(f_base, sr) + generate_sine(f_base * 2**(1/12), sr)
    sig_good = generate_sine(f_base, sr) + generate_sine(f_base * 2, sr)
    
    _, int_bad = timbral_dissonance(sig_bad, sr)
    _, int_good = timbral_dissonance(sig_good, sr)
    
    assert int_bad > int_good

def test_plomp_levelt_curve_shape():
    """Verify that the computed dissonance curve vs frequency interval matches Sethares theory."""
    sr = 22050
    base_freq = 500.0
    ratios = np.linspace(1.0, 2.2, 20) # Coarse sweep for speed
    
    vals_comp = []
    vals_theo = []
    
    for r in ratios:
        f2 = base_freq * r
        sig = generate_sine(base_freq, sr, 0.2) + generate_sine(f2, sr, 0.2)
        _, integral = timbral_dissonance(sig, sr, n_fft=32768)
        vals_comp.append(integral)
        
        # Theory
        f_diff = abs(f2 - base_freq)
        f_min = base_freq
        s = f_diff / (0.021 * f_min + 19.0)
        d = np.exp(-3.5 * s) - np.exp(-5.75 * s)
        vals_theo.append(d)
        
    corr = np.corrcoef(vals_comp, vals_theo)[0,1]
    # We expect positive correlation. 
    # At ratio 1.0 (unison), theory=0, computed=small(leakage).
    # At max dissonance (~ratio 1.05), both peak.
    # Then both drop.
    assert corr > 0.85, f"Dissonance curve correlation {corr} is too low"
