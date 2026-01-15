"""
Tests to verify that differentiable filter implementations match SciPy's output.
"""
import pytest
import torch
import numpy as np
import scipy.signal

from dissonance_app.differentiable_filters import (
    diff_butter_lowpass_sos,
    diff_butter_bandpass_sos,
)


class TestLowpassFilterConsistency:
    """Verify lowpass filters match SciPy."""
    
    @pytest.mark.parametrize("cutoff,fs", [
        (100.0, 22050),
        (500.0, 44100),
        (1000.0, 48000),
        (50.0, 16000),
    ])
    def test_lowpass_order2_matches_scipy(self, cutoff, fs):
        """Lowpass order 2 should match SciPy exactly."""
        scipy_sos = scipy.signal.butter(2, cutoff, btype='low', fs=fs, output='sos')
        diff_sos = diff_butter_lowpass_sos(cutoff, fs, order=2).numpy()
        
        assert scipy_sos.shape == diff_sos.shape, f"Shape mismatch: {scipy_sos.shape} vs {diff_sos.shape}"
        np.testing.assert_allclose(scipy_sos, diff_sos, rtol=1e-5, atol=1e-7,
                                   err_msg="Lowpass order 2 coefficients don't match SciPy")


class TestBandpassFilterConsistency:
    """Verify bandpass filters match SciPy."""
    
    @pytest.mark.parametrize("low,high,fs", [
        (20.0, 200.0, 22050),
        (100.0, 500.0, 44100),
        (50.0, 150.0, 16000),
        (30.0, 70.0, 8000),
    ])
    def test_bandpass_order4_nongrad_matches_scipy(self, low, high, fs):
        """Bandpass order 4 (non-gradient mode) should use SciPy fallback and match exactly."""
        scipy_sos = scipy.signal.butter(4, [low, high], btype='band', fs=fs, output='sos')
        # Non-tensor input triggers SciPy fallback
        diff_sos = diff_butter_bandpass_sos(low, high, fs, order=4).numpy()
        
        assert scipy_sos.shape == diff_sos.shape, f"Shape mismatch: {scipy_sos.shape} vs {diff_sos.shape}"
        np.testing.assert_allclose(scipy_sos, diff_sos, rtol=1e-5, atol=1e-7,
                                   err_msg="Bandpass order 4 (non-grad) coefficients don't match SciPy")
    
    @pytest.mark.parametrize("low,high,fs", [
        (20.0, 200.0, 22050),
        (100.0, 500.0, 44100),
    ])
    def test_bandpass_order4_grad_has_correct_shape(self, low, high, fs):
        """Bandpass order 4 (gradient mode) should have correct shape (4 sections)."""
        low_t = torch.tensor(low, requires_grad=True)
        high_t = torch.tensor(high, requires_grad=True)
        diff_sos = diff_butter_bandpass_sos(low_t, high_t, fs, order=4)
        
        # Bandpass order 4 should have 4 sections (2*N poles)
        assert diff_sos.shape == (4, 6), f"Expected shape (4, 6), got {diff_sos.shape}"
    
    @pytest.mark.parametrize("low,high,fs", [
        (20.0, 200.0, 22050),
        (100.0, 500.0, 44100),
    ])
    def test_bandpass_order4_grad_gradients_flow(self, low, high, fs):
        """Bandpass order 4 (gradient mode) should have gradients that flow back."""
        low_t = torch.tensor(low, requires_grad=True)
        high_t = torch.tensor(high, requires_grad=True)
        diff_sos = diff_butter_bandpass_sos(low_t, high_t, fs, order=4)
        
        # Compute a simple loss and backprop
        loss = diff_sos.sum()
        loss.backward()
        
        assert low_t.grad is not None, "Gradient should flow to low_freq"
        assert high_t.grad is not None, "Gradient should flow to high_freq"
        assert not torch.isnan(low_t.grad), "low_freq gradient should not be NaN"
        assert not torch.isnan(high_t.grad), "high_freq gradient should not be NaN"


class TestFilteredSignalConsistency:
    """Verify that applying filters produces similar results."""
    
    def test_lowpass_filter_application_matches(self):
        """Filtered signal using diff filters should match SciPy-filtered signal."""
        fs = 22050
        cutoff = 100.0
        
        # Generate test signal
        t = np.linspace(0, 1, fs)
        signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 500 * t)
        
        # SciPy filtering
        scipy_sos = scipy.signal.butter(2, cutoff, btype='low', fs=fs, output='sos')
        scipy_filtered = scipy.signal.sosfilt(scipy_sos, signal)
        
        # Differentiable filtering (non-grad path should be identical)
        diff_sos = diff_butter_lowpass_sos(cutoff, fs, order=2).numpy()
        diff_filtered = scipy.signal.sosfilt(diff_sos, signal)
        
        # Note: Slightly looser tolerance due to float32 vs float64 accumulation
        np.testing.assert_allclose(scipy_filtered, diff_filtered, rtol=1e-4, atol=1e-4,
                                   err_msg="Filtered signals should match")
    
    def test_bandpass_filter_application_matches(self):
        """Filtered signal using diff bandpass should match SciPy-filtered signal."""
        fs = 22050
        low, high = 80.0, 120.0
        
        # Generate test signal with components in and out of band
        t = np.linspace(0, 1, fs)
        signal = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 500 * t) + np.sin(2 * np.pi * 20 * t)
        
        # SciPy filtering
        scipy_sos = scipy.signal.butter(4, [low, high], btype='band', fs=fs, output='sos')
        scipy_filtered = scipy.signal.sosfilt(scipy_sos, signal)
        
        # Differentiable filtering (non-grad path should use SciPy fallback)
        diff_sos = diff_butter_bandpass_sos(low, high, fs, order=4).numpy()
        diff_filtered = scipy.signal.sosfilt(diff_sos, signal)
        
        # Note: Slightly looser tolerance due to float32 vs float64 accumulation
        np.testing.assert_allclose(scipy_filtered, diff_filtered, rtol=1e-4, atol=1e-4,
                                   err_msg="Bandpass filtered signals should match")
