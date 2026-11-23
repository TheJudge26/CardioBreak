"""Milestone 3: Red-Team Adversarial Pulse Injection.

Starter skeleton for implementing:
- Synthetic pulse generators:
  * Variant A: Rigid Sine (constant BPM)
  * Variant B: HRV-Modulated drift
  * Variant C: Per-patch phase and amplitude jitter
- 8-bit TPDF dithering to preserve sub-LSB signals
- Feathered skin alpha matte compositing
- Measuring attack bypass rate on Milestone 2 detector
"""

import numpy as np


def apply_tpdf_dithering(frame_float: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
    """Apply Triangular Probability Density Function (TPDF) noise and round to uint8.
    
    TODO:
    1. Generate u1, u2 ~ Uniform(-0.5, 0.5).
    2. Compute d = u1 - u2 (symmetric triangular in [-1.0, 1.0]).
    3. Add noise to floating frame: round(frame_float + d) and clip to [0, 255].
    """
    # TODO: Implement TPDF dithering
    raise NotImplementedError("Implement apply_tpdf_dithering")


def generate_variant_a(duration_sec: float, fs: float = 30.0, bpm: float = 75.0) -> np.ndarray:
    """Variant A: Rigid Sine pulse signal.
    
    TODO:
    Generate s(t) = A * sin(2 * pi * f0 * t) with f0 = bpm / 60.
    """
    # TODO: Implement Variant A
    raise NotImplementedError("Implement generate_variant_a")


def generate_variant_b(duration_sec: float, fs: float = 30.0, bpm: float = 75.0) -> np.ndarray:
    """Variant B: HRV-Modulated pulse signal.
    
    TODO:
    Add low-frequency instantaneous frequency drift Delta_f_HRV(t) and integrate phase.
    """
    # TODO: Implement Variant B
    raise NotImplementedError("Implement generate_variant_b")


def generate_variant_c(duration_sec: float, roi_names: list, fs: float = 30.0, bpm: float = 75.0) -> dict:
    """Variant C: Per-patch phase and amplitude jitter.
    
    TODO:
    Generate separate pulse signal for each facial skin ROI with small smooth
    phase offsets (delta_phi_k) and amplitude scaling factors (a_k).
    """
    # TODO: Implement Variant C
    raise NotImplementedError("Implement generate_variant_c")


def inject_pulse_into_video(video_path: str, output_path: str, variant: str = "variant_b"):
    """Inject synthetic pulse into a fake video and export losslessly (FFV1 rgb24 or PNG sequence).
    
    TODO:
    1. Read input video frames.
    2. Extract facial skin alpha matte and apply Gaussian feathering.
    3. Multiply pulse modulation by hemoglobin RGB weights [0.3, 1.0, 0.3].
    4. Add TPDF dithering and save losslessly without re-encoding to H.264.
    """
    # TODO: Implement pulse injection
    raise NotImplementedError("Implement inject_pulse_into_video")


def run_m3():
    """TODO: Inject synthetic pulses into true-positive fakes and re-test on Milestone 2 detector."""
    print("Running Milestone 3...")
    # TODO: Add your attack code here


if __name__ == "__main__":
    run_m3()
