"""
Multi-source synthetic sound generator for roughness model training.

Features:
- Reference pitch from Beta(7,7) over piano keyboard
- 1-4 random sources per sample
- Each source: random fundamental ±0.5 octave from reference
- Random polynomial timbre (1/n based)
- Inharmonicity: log-normal shifts per partial (~10 cents stdev)
- Random amplitude per source
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple


def sample_reference_pitch() -> float:
    """
    Sample a reference pitch from piano range using Beta(7,7).
    Returns frequency in Hz.
    """
    b = np.random.beta(7, 7)
    midi_note = 21 + 87 * b  # A0 (21) to C8 (108)
    return 440 * (2 ** ((midi_note - 69) / 12))


def sample_fundamental_near_reference(ref_hz: float, octave_spread: float = 0.5) -> float:
    """
    Sample a fundamental within ±octave_spread octaves of reference.
    """
    # Uniform in log-frequency space
    log_shift = np.random.uniform(-octave_spread, octave_spread)
    return ref_hz * (2 ** log_shift)


def sample_polynomial_timbre(n_harmonics: int = 16, n_coeffs: int = 3) -> np.ndarray:
    """
    Generate a random amplitude profile using polynomial in 1/n.
    
    Profile: amp[n] = sum_k(c_k / n^k) where k=1,2,...,n_coeffs
    Coefficients are sampled to ensure reasonable decay.
    
    Returns normalized amplitude array for harmonics 1..n_harmonics.
    """
    # Sample coefficients: c_1 dominant, higher powers smaller
    coeffs = []
    for k in range(1, n_coeffs + 1):
        # Coefficient range decreases with power
        c = np.random.uniform(0.5, 2.0) if k == 1 else np.random.uniform(-0.5, 1.0)
        coeffs.append(c)
    
    # Compute amplitudes
    amps = np.zeros(n_harmonics)
    for n in range(1, n_harmonics + 1):
        for k, c in enumerate(coeffs, start=1):
            amps[n-1] += c / (n ** k)
    
    # Ensure positive and normalize
    amps = np.maximum(amps, 0.01)
    amps = amps / amps[0]  # Normalize so fundamental = 1
    
    return amps


def sample_inharmonicity(n_harmonics: int, stdev_cents: float = 10.0) -> np.ndarray:
    """
    Sample log-frequency shifts for each harmonic.
    
    Returns array of log-frequency shifts (multiply freq by exp(shift)).
    stdev_cents: standard deviation in cents (100 cents = 1 semitone)
    """
    # Convert cents to log-frequency: 1 cent = log(2)/1200
    stdev_log = stdev_cents * np.log(2) / 1200
    
    # Sample shifts (fundamental stays fixed at 0)
    shifts = np.zeros(n_harmonics)
    shifts[1:] = np.random.normal(0, stdev_log, n_harmonics - 1)
    
    return shifts


class SyntheticSource:
    """A single harmonic source with timbre and inharmonicity."""
    
    def __init__(
        self,
        fundamental: float,
        amplitude: float,
        timbre_amps: np.ndarray,
        inharmonicity: np.ndarray
    ):
        self.fundamental = fundamental
        self.amplitude = amplitude
        self.timbre_amps = timbre_amps
        self.inharmonicity = inharmonicity
        self.n_harmonics = len(timbre_amps)
    
    def get_partials(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (frequencies, amplitudes) for all partials."""
        freqs = np.array([
            self.fundamental * n * np.exp(self.inharmonicity[n-1])
            for n in range(1, self.n_harmonics + 1)
        ])
        amps = self.amplitude * self.timbre_amps
        return freqs, amps
    
    def generate_signal(self, t: torch.Tensor, device: str = 'cpu') -> torch.Tensor:
        """Generate the audio signal for this source."""
        freqs, amps = self.get_partials()
        sig = torch.zeros_like(t)
        for f, a in zip(freqs, amps):
            sig = sig + a * torch.sin(2 * torch.pi * f * t)
        return sig


class MultiSourceSample:
    """A complete synthetic sample with multiple sources."""
    
    def __init__(
        self,
        sources: List[SyntheticSource],
        reference_pitch: float,
        sr: int = 22050,
        duration: float = 0.5
    ):
        self.sources = sources
        self.reference_pitch = reference_pitch
        self.sr = sr
        self.duration = duration
    
    def generate_signal(self, device: str = 'cpu') -> torch.Tensor:
        """Generate combined audio signal."""
        t = torch.linspace(0, self.duration, int(self.sr * self.duration), device=device)
        sig = torch.zeros_like(t)
        for source in self.sources:
            sig = sig + source.generate_signal(t, device)
        return sig
    
    def compute_sethares_dissonance(self, alpha: float = 0.021, beta: float = 19.0) -> float:
        """
        Compute total Sethares dissonance for all partials across all sources.
        """
        # Collect all partials
        all_freqs = []
        all_amps = []
        for source in self.sources:
            freqs, amps = source.get_partials()
            all_freqs.extend(freqs)
            all_amps.extend(amps)
        
        all_freqs = np.array(all_freqs)
        all_amps = np.array(all_amps)
        n = len(all_freqs)
        
        # Compute pairwise dissonance
        total_d = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                f1, f2 = all_freqs[i], all_freqs[j]
                a1, a2 = all_amps[i], all_amps[j]
                
                f_min = min(f1, f2)
                f_diff = abs(f1 - f2)
                s = f_diff / (alpha * f_min + beta)
                d = a1 * a2 * (np.exp(-3.5 * s) - np.exp(-5.75 * s))
                total_d += max(0, d)
        
        return total_d


def generate_random_sample(
    n_harmonics: int = 16,
    sr: int = 22050,
    duration: float = 0.5,
    min_sources: int = 1,
    max_sources: int = 4,
    octave_spread: float = 0.5,
    inharmonicity_cents: float = 10.0
) -> MultiSourceSample:
    """
    Generate a random multi-source synthetic sample.
    """
    # Reference pitch
    ref_pitch = sample_reference_pitch()
    
    # Number of sources
    n_sources = np.random.randint(min_sources, max_sources + 1)
    
    sources = []
    for _ in range(n_sources):
        # Fundamental near reference
        fundamental = sample_fundamental_near_reference(ref_pitch, octave_spread)
        
        # Random amplitude (log-uniform between 0.3 and 1.0)
        amplitude = np.exp(np.random.uniform(np.log(0.3), np.log(1.0)))
        
        # Random timbre
        timbre = sample_polynomial_timbre(n_harmonics)
        
        # Random inharmonicity
        inharm = sample_inharmonicity(n_harmonics, inharmonicity_cents)
        
        sources.append(SyntheticSource(fundamental, amplitude, timbre, inharm))
    
    return MultiSourceSample(sources, ref_pitch, sr, duration)


def plot_sample_diagnostics(sample: MultiSourceSample, save_path: str = None):
    """
    Plot spectrogram, source info, Sethares curve, and save WAV.
    """
    import scipy.io.wavfile as wavfile
    from scipy import signal as scipy_signal
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    
    # Generate signal
    sig = sample.generate_signal().numpy()
    
    # Normalize and save WAV
    if save_path:
        wav_path = save_path.replace('.png', '.wav')
        sig_normalized = sig / (np.abs(sig).max() + 1e-8) * 0.9
        wavfile.write(wav_path, sample.sr, (sig_normalized * 32767).astype(np.int16))
        print(f"Saved WAV to {wav_path}")
    
    # 1. Waveform
    ax = axes[0, 0]
    t = np.linspace(0, sample.duration, len(sig))
    ax.plot(t, sig)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Waveform ({len(sample.sources)} sources)')
    ax.grid(alpha=0.3)
    
    # 2. Spectrogram
    ax = axes[0, 1]
    f, t_spec, Sxx = scipy_signal.spectrogram(sig, sample.sr, nperseg=2048, noverlap=1024)
    ax.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='magma')
    ax.set_ylim(0, 5000)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Spectrogram')
    
    # 3. Partial frequencies and amplitudes
    ax = axes[1, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(sample.sources)))
    for i, source in enumerate(sample.sources):
        freqs, amps = source.get_partials()
        ax.scatter(freqs, amps, c=[colors[i]], s=50, alpha=0.7, 
                   label=f'Source {i+1}: f0={source.fundamental:.0f}Hz')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Partial Frequencies & Amplitudes')
    ax.set_xlim(0, 5000)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Timbre profiles
    ax = axes[1, 1]
    for i, source in enumerate(sample.sources):
        ax.plot(range(1, len(source.timbre_amps) + 1), source.timbre_amps, 
                'o-', color=colors[i], label=f'Source {i+1}')
    ax.set_xlabel('Harmonic Number')
    ax.set_ylabel('Relative Amplitude')
    ax.set_title('Timbre Profiles')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 5. Plomp-Levelt style Sethares curve
    ax = axes[2, 0]
    # Compute dissonance for ratios 1.0 to 2.5 relative to first source fundamental
    ref_f0 = sample.sources[0].fundamental
    ratios = np.linspace(1.0, 2.5, 200)
    dissonances = []
    
    for ratio in ratios:
        # Create a virtual second tone at ref_f0 * ratio with same timbre as source 0
        virtual_freqs = []
        virtual_amps = []
        for n in range(1, sample.sources[0].n_harmonics + 1):
            virtual_freqs.append(ref_f0 * ratio * n)
            virtual_amps.append(sample.sources[0].timbre_amps[n-1])
        
        # Compute dissonance between source 0 and virtual tone
        source_freqs, source_amps = sample.sources[0].get_partials()
        total_d = 0.0
        for i, (f1, a1) in enumerate(zip(source_freqs, source_amps)):
            for j, (f2, a2) in enumerate(zip(virtual_freqs, virtual_amps)):
                f_min = min(f1, f2)
                f_diff = abs(f1 - f2)
                s = f_diff / (0.021 * f_min + 19.0)
                d = a1 * a2 * (np.exp(-3.5 * s) - np.exp(-5.75 * s))
                total_d += max(0, d)
        dissonances.append(total_d)
    
    ax.plot(ratios, dissonances, 'b-', lw=2)
    ax.axvline(1.5, color='r', linestyle='--', alpha=0.5, label='1.5 (Fifth)')
    ax.axvline(2.0, color='g', linestyle='--', alpha=0.5, label='2.0 (Octave)')
    ax.axvline(4/3, color='orange', linestyle='--', alpha=0.5, label='4/3 (Fourth)')
    ax.set_xlabel('Frequency Ratio')
    ax.set_ylabel('Dissonance')
    ax.set_title(f'Sethares Curve (Source 1, f0={ref_f0:.0f}Hz)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 6. Source info text
    ax = axes[2, 1]
    ax.axis('off')
    info_text = f"Reference Pitch: {sample.reference_pitch:.1f} Hz\n\n"
    info_text += "Sources:\n"
    for i, source in enumerate(sample.sources):
        ratio_to_ref = source.fundamental / sample.reference_pitch
        cents_from_ref = 1200 * np.log2(ratio_to_ref)
        info_text += f"  {i+1}. f0={source.fundamental:.1f}Hz ({cents_from_ref:+.0f}¢), amp={source.amplitude:.2f}\n"
    info_text += f"\nTotal Sethares Dissonance: {sample.compute_sethares_dissonance():.4f}"
    ax.text(0.1, 0.5, info_text, transform=ax.transAxes, fontsize=12, 
            verticalalignment='center', fontfamily='monospace')
    ax.set_title('Sample Information')
    
    # Sethares dissonance
    diss = sample.compute_sethares_dissonance()
    fig.suptitle(f'Multi-Source Synthetic Sample | Total Dissonance: {diss:.4f}', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Generate and plot a few random samples
    np.random.seed(42)
    
    for i in range(4):
        sample = generate_random_sample()
        plot_sample_diagnostics(sample, f'synth_sample_{i+1}.png')
        print(f"Sample {i+1}: {len(sample.sources)} sources, "
              f"ref={sample.reference_pitch:.0f}Hz, "
              f"dissonance={sample.compute_sethares_dissonance():.4f}")
