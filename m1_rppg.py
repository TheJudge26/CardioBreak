"""Milestone 1: rPPG Pulse Extraction & Validation (UBFC-rPPG).

Starter skeleton for implementing:
- 4th-order zero-phase Butterworth bandpass filter (0.7-3.0 Hz)
- Green, CHROM, and POS extraction algorithms
- FFT spectral peak BPM estimation
- Mean Absolute Error (MAE) calculation against ground truth
"""

import numpy as np
from scipy import signal


def filter_rppg(sig: np.ndarray, fs: float = 30.0, low_hz: float = 0.7, high_hz: float = 3.0) -> np.ndarray:

    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")

    if not (0 < low_hz < high_hz < fs / 2):
        raise ValueError("Invalid frequency range for the given sampling frequency")

    sig = signal.detrend(sig, type = "linear")
    sos = signal.butter(4, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    filtered = signal.sosfiltfilt(sos, sig)
    return filtered


# t = np.linspace(0, 10, 300)

# heartbeat = np.sin(2 * np.pi * 1.2 * t)

# noise = (0.5 * np.sin(2 * np.pi * 0.2 * t) + 0.5 * np.sin(2 * np.pi * 4.0 * t))

# test_signal = heartbeat + noise

# result = filter_rppg(test_signal)

# print("Original signal: ", test_signal.shape)
# print("Filtered signal: ", result.shape)

# print("Original first 10:", test_signal[:10])
# print("Filtered first 10:", result[:10])


def extract_green(rgb_traces: np.ndarray, fs: float = 30.0) -> np.ndarray:
    green = rgb_traces [:, 1]
    pulse = filter_rppg(green, fs)
    return pulse


def extract_chrom(rgb_traces: np.ndarray, fs: float = 30.0, win_sec: float = 1.6) -> np.ndarray:

    win_samples = int(round(win_sec * fs))
    if win_samples < 2:
        raise ValueError("CHROM window is too short")
    if len(rgb_traces) < win_samples:
        raise ValueError("Video is too short for a CHROM window")

    step = win_samples // 2

    pulse_sum = np.zeros(len(rgb_traces))
    weight_sum = np.zeros(len(rgb_traces))

    starts = list(range(0, len(rgb_traces) - win_samples + 1, step))

    last_start = len(rgb_traces) - win_samples

    if starts[-1] != last_start:
        starts.append(last_start)

    for start in starts:
        end = start + win_samples
        window = rgb_traces[start:end]
        R = window[:, 0]
        G = window[:, 1]
        B = window[:, 2]

        Xs = 3 * R - 2 * G
        Ys = 1.5 * R + G - 1.5 * B

        filtered_Xs = filter_rppg(Xs, fs)
        filtered_Ys = filter_rppg(Ys, fs)

        if np.std(filtered_Ys) < 1e-8:
            alpha = 0.0
        else:
            alpha = np.std(filtered_Xs) / np.std(filtered_Ys)

        window_pulse = filtered_Xs - alpha * filtered_Ys
        pulse_sum[start:end] += window_pulse
        weight_sum[start:end] += 1

    if np.any(weight_sum == 0):
        raise ValueError("Some frames were not covered by CHROM windows")

    pulse = pulse_sum / weight_sum
    pulse = filter_rppg(pulse, fs)
    return pulse

    # TODO:
    # 2. Bandpass filter Xs and Ys.
    # 3. Calculate alpha = std(Xs) / std(Ys).
    # 4. Pulse = Xs - alpha * Ys.


def extract_pos(rgb_traces: np.ndarray, fs: float = 30.0, win_sec: float = 1.6) -> np.ndarray:
    win_samples = int(round(win_sec * fs))

    if win_samples < 2:
        raise ValueError("POS window is too short")
    if len(rgb_traces) < win_samples:
        raise ValueError("Video is too short for a POS window")

    step = win_samples // 2

    pulse_sum = np.zeros(len(rgb_traces))
    weight_sum = np.zeros(len(rgb_traces))

    starts = list(range(0, len(rgb_traces) - win_samples + 1, step))

    last_start = len(rgb_traces) - win_samples

    if starts[-1] != last_start:
        starts.append(last_start)

    for start in starts:
        end = start + win_samples
        window = rgb_traces[start:end]

        R = window[:, 0]
        G = window[:, 1]
        B = window[:, 2]

        R_norm = (R / np.mean(R)) - 1
        G_norm = (G / np.mean(G)) - 1
        B_norm = (B / np.mean(B)) - 1
        S1 = G_norm - B_norm
        S2 = -2 * R_norm + G_norm + B_norm

        std_S1 = np.std(S1)
        std_S2 = np.std(S2)

        if std_S2 < 1e-8:
            alpha = 0.0
        else:
            alpha = std_S1 / std_S2

        window_pulse = S1 + alpha * S2

        pulse_sum[start:end] += window_pulse
        weight_sum[start:end] += 1

    if np.any(weight_sum == 0):
        raise ValueError("Some frames were not covered by POS windows")

    pulse = pulse_sum / weight_sum
    filtered_pulse = filter_rppg(pulse, fs)

    return filtered_pulse

    # TODO:
    # 1. Normalize temporal RGB traces by their mean.
    # 2. Project onto skin-orthogonal plane:
    #    S1 = G_norm - B_norm
    #    S2 = -2*R_norm + G_norm + B_norm
    # 3. Compute h = S1 + (std(S1)/std(S2)) * S2.
    # 4. Combine windows via overlap-add and bandpass filter.



def estimate_bpm_fft(pulse: np.ndarray, fs: float = 30.0) -> float:

    if len(pulse) == 0:
        raise ValueError("Pulse signal cannot be empty")

    window = np.hanning(len(pulse))
    windowed_pulse = pulse * window


    frequencies = np.fft.rfftfreq(len(pulse), d=1/fs)
    magnitudes = np.abs(np.fft.rfft(windowed_pulse))
    valid = (frequencies >= 0.7) & (frequencies <= 3.0)

    if not np.any(valid):
        raise ValueError("No FFT frequency bins within the physiological range")

    valid_frequencies = frequencies[valid]
    valid_magnitudes = magnitudes[valid]

    peak_index = np.argmax(valid_magnitudes)
    peak_frequency = valid_frequencies[peak_index]

    bpm = peak_frequency * 60

    return float(bpm)
    # TODO:
    # 1. Apply Hann window to pulse signal.
    # 2. Compute rfft and frequency bins.
    # 3. Locate peak within physiological range: [0.7 Hz (42 BPM), 3.0 Hz (180 BPM)].
    # 4. Return BPM = f_peak * 60.


def run_m1():
    """TODO: Load UBFC-rPPG video and ground truth, compute MAE, and check gate (MAE <= 5.0 BPM)."""
    print("Running Milestone 1...")
    # TODO: Add your validation code here


if __name__ == "__main__":
    fs = 30.0
    duration = 30
    expected_bpm = 72

    # Generate a synthetic heartbeat
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    heartbeat = np.sin(2 * np.pi * 1.2 * t)

    # Generate synthetic RGB traces
    R = 100 + 0.5 * heartbeat
    G = 120 + 1.5 * heartbeat
    B = 90 + 0.3 * heartbeat

    rgb_traces = np.column_stack((R, G, B))

    # Run all three extraction algorithms
    green_signal = extract_green(rgb_traces, fs)
    chrom_signal = extract_chrom(rgb_traces, fs)
    pos_signal = extract_pos(rgb_traces, fs)

    # Estimate BPM for each algorithm
    green_bpm = estimate_bpm_fft(green_signal, fs)
    chrom_bpm = estimate_bpm_fft(chrom_signal, fs)
    pos_bpm = estimate_bpm_fft(pos_signal, fs)

    # Compare results
    results = {
        "Green": green_bpm,
        "CHROM": chrom_bpm,
        "POS": pos_bpm
    }

    print("\n========== M1 SYNTHETIC TEST ==========")
    print(f"Expected BPM: {expected_bpm}\n")

    for algorithm, estimated_bpm in results.items():
        error = abs(expected_bpm - estimated_bpm)

        print(f"{algorithm}:")
        print(f"  Estimated BPM: {estimated_bpm:.2f}")
        print(f"  Absolute Error: {error:.2f} BPM\n")


    # run_m1()
