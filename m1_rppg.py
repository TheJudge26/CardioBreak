"""Milestone 1: rPPG Pulse Extraction & Validation (UBFC-rPPG).

Starter skeleton for implementing:
- 4th-order zero-phase Butterworth bandpass filter (0.7-3.0 Hz)
- Green, CHROM, and POS extraction algorithms
- FFT spectral peak BPM estimation
- Mean Absolute Error (MAE) calculation against ground truth
"""

import numpy as np


def filter_rppg(sig: np.ndarray, fs: float = 30.0, low_hz: float = 0.7, high_hz: float = 3.0) -> np.ndarray:
    """Detrend and apply 4th-order zero-phase Butterworth bandpass filter.
    
    TODO:
    1. Linearly detrend signal (scipy.signal.detrend).
    2. Design 4th-order Butterworth bandpass filter using Second-Order Sections (sos).
    3. Apply zero-phase filtering (scipy.signal.sosfiltfilt).
    """
    # TODO: Implement filtering
    raise NotImplementedError("Implement filter_rppg")


def extract_green(rgb_traces: np.ndarray, fs: float = 30.0) -> np.ndarray:
    """Extract pulse signal from spatial Green channel mean.
    
    TODO:
    Extract the Green channel from rgb_traces and filter it using filter_rppg().
    """
    # TODO: Implement Green extraction
    raise NotImplementedError("Implement extract_green")


def extract_chrom(rgb_traces: np.ndarray, fs: float = 30.0) -> np.ndarray:
    """Chrominance-based rPPG extraction (de Haan & Jeanne 2013).
    
    TODO:
    1. Project RGB onto chrominance signals:
       Xs = 3*R - 2*G
       Ys = 1.5*R + G - 1.5*B
    2. Bandpass filter Xs and Ys.
    3. Calculate alpha = std(Xs) / std(Ys).
    4. Pulse = Xs - alpha * Ys.
    """
    # TODO: Implement CHROM
    raise NotImplementedError("Implement extract_chrom")


def extract_pos(rgb_traces: np.ndarray, fs: float = 30.0, win_sec: float = 1.6) -> np.ndarray:
    """Plane-Orthogonal-to-Skin (POS) rPPG extraction (Wang et al. 2017).
    
    TODO:
    1. Normalize temporal RGB traces by their mean.
    2. Project onto skin-orthogonal plane:
       S1 = G_norm - B_norm
       S2 = -2*R_norm + G_norm + B_norm
    3. Compute h = S1 + (std(S1)/std(S2)) * S2.
    4. Combine windows via overlap-add and bandpass filter.
    """
    # TODO: Implement POS
    raise NotImplementedError("Implement extract_pos")


def estimate_bpm_fft(pulse: np.ndarray, fs: float = 30.0) -> float:
    """Estimate Heart Rate in BPM using real FFT spectral peak in [42, 180] BPM.
    
    TODO:
    1. Apply Hann window to pulse signal.
    2. Compute rfft and frequency bins.
    3. Locate peak within physiological range: [0.7 Hz (42 BPM), 3.0 Hz (180 BPM)].
    4. Return BPM = f_peak * 60.
    """
    # TODO: Implement FFT peak BPM estimation
    raise NotImplementedError("Implement estimate_bpm_fft")


def run_m1():
    """TODO: Load UBFC-rPPG video and ground truth, compute MAE, and check gate (MAE <= 5.0 BPM)."""
    print("Running Milestone 1...")
    # TODO: Add your validation code here


if __name__ == "__main__":
    run_m1()
