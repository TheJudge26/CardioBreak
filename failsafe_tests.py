"""
Fail-Safe and Signal Processing Tests for Milestone 1 (rPPG)

This file tests the input validation, fail-safe mechanisms, and signal
processing behavior implemented in m1_rppg.py.

It covers:
- Empty signals and invalid RGB trace inputs.
- Invalid input dimensions, shapes, and non-finite values.
- Invalid sampling frequencies.
- Invalid CHROM and POS window durations and lengths.
- Videos that are too short for CHROM and POS processing.
- Zero RGB channel means during POS processing.
- FFT estimation accuracy across different signal durations.
- FFT estimation accuracy under varying noise levels.
- FFT behavior when multiple frequency components are present.

The tests verify that invalid inputs raise the expected ValueError
and evaluate how reliably the signal processing functions estimate
heart rate under different synthetic signal conditions.

Run this file independently using: python failsafe_tests.py
"""

import numpy as np

from m1_rppg import (
    validate_rgb_traces,
    filter_rppg,
    extract_green,
    extract_chrom,
    extract_pos,
    estimate_bpm_fft,
)

np.random.seed(42)


# ==========================================
# GLOBAL VARIABLES
# ==========================================

fs = 30.0
duration = 10

t = np.arange(int(duration * fs)) / fs

# Valid RGB traces
rgb_traces = np.array([
    [120.0, 130.0, 110.0],
    [121.0, 131.0, 111.0],
    [122.0, 132.0, 112.0],
    [123.0, 133.0, 113.0]
])

# Valid synthetic signal
sig = (
    120
    + 2 * np.sin(2 * np.pi * 1.2 * t)
    + 0.5 * np.sin(2 * np.pi * 5 * t)
)

# Invalid signals
invalid_signals = [
    np.array([]),
    np.array([[1.0, 2.0], [3.0, 4.0]]),
    np.array([1.0, 2.0, np.nan, 4.0]),
    np.array([1.0, 2.0, np.inf, 4.0])
]

# Invalid RGB traces
invalid_rgb_traces = [
    np.array([]),
    np.array([[1.0, 2.0], [3.0, 4.0]]),
    np.array([[1.0, np.nan, 3.0]]),
    np.array([[1.0, np.inf, 3.0]])
]

# Invalid sampling frequencies
invalid_fs_values = [0, -30, np.nan, np.inf]

# Invalid window durations
invalid_win_sec = [0, -1.0, np.nan, np.inf]

# Window durations producing fewer than 2 samples
short_win_values = [0.01, 0.02]

# RGB traces shorter than a 1.6-second window at 30 FPS
short_rgb_traces = np.full((20, 3), 120.0)

# POS RGB traces with a zero-mean red channel
zero_mean_rgb_traces = np.column_stack((
    np.zeros(48),
    np.full(48, 120.0),
    np.full(48, 110.0)
))


# ==========================================
# GLOBAL TESTING HELPER
# ==========================================

def run_invalid_input_tests(function, test_cases):

    for description, args in test_cases:

        try:
            function(*args)
        except ValueError as error:
            print(f"PASS: {description} -> {error}")
        else:
            print(f"FAIL: {description} was accepted")


# ==========================================
# TESTING FUNCTIONS
# ==========================================


# ==========================================
# SIGNAL VALIDATION TESTING
# ==========================================

def signal_testing():

    test_cases = [
        (f"Invalid signal #{i + 1}", (invalid_sig, fs))
        for i, invalid_sig in enumerate(invalid_signals)
    ]

    run_invalid_input_tests(filter_rppg, test_cases)


# ==========================================
# SAMPLING FREQUENCY TESTING
# ==========================================

def sample_testing():

    test_cases = [
        (f"Invalid fs={invalid_fs}", (sig, invalid_fs))
        for invalid_fs in invalid_fs_values
    ]

    run_invalid_input_tests(filter_rppg, test_cases)


# ==========================================
# GREEN SIGNAL VALIDATION TESTING
# ==========================================

def green_testing():

    test_cases = [
        (f"Invalid RGB traces #{i + 1}", (invalid_rgb, fs))
        for i, invalid_rgb in enumerate(invalid_rgb_traces)
    ]

    run_invalid_input_tests(extract_green, test_cases)


# ==========================================
# GREEN SAMPLING FREQUENCY TESTING
# ==========================================

def green_sample_testing():

    test_cases = [
        (f"Invalid fs={invalid_fs}", (rgb_traces, invalid_fs))
        for invalid_fs in invalid_fs_values
    ]

    run_invalid_input_tests(extract_green, test_cases)


# ==========================================
# CHROM RGB VALIDATION TESTING
# ==========================================

def chrom_testing():

    test_cases = [
        (f"Invalid RGB traces #{i + 1}", (invalid_rgb, fs))
        for i, invalid_rgb in enumerate(invalid_rgb_traces)
    ]

    run_invalid_input_tests(extract_chrom, test_cases)

# ==========================================
# CHROM SAMPLING FREQUENCY TESTING
# ==========================================

def chrom_freq_testing():

    test_cases = [
        (f"Invalid fs={invalid_fs}", (rgb_traces, invalid_fs))
        for invalid_fs in invalid_fs_values
    ]

    run_invalid_input_tests(extract_chrom, test_cases)


# ==========================================
# CHROM WINDOW DURATION TESTING
# ==========================================

def chrom_window_duration_testing():

    test_cases = [
        (f"Invalid win_sec={invalid_win}", (rgb_traces, fs, invalid_win))
        for invalid_win in invalid_win_sec
    ]

    run_invalid_input_tests(extract_chrom, test_cases)


# ==========================================
# CHROM MINIMUM WINDOW LENGTH TESTING
# ==========================================

def chrom_window_length_testing():

    test_cases = [
        (f"Short window={short_win}", (rgb_traces, fs, short_win))
        for short_win in short_win_values
    ]

    run_invalid_input_tests(extract_chrom, test_cases)


# ==========================================
# CHROM VIDEO LENGTH TESTING
# ==========================================

def chrom_video_length_testing():

    test_cases = [
        ("Video shorter than CHROM window", (short_rgb_traces, fs, 1.6))
    ]

    run_invalid_input_tests(extract_chrom, test_cases)


# ==========================================
# POS RGB VALIDATION TESTING
# ==========================================

def pos_testing():

    test_cases = [
        (f"Invalid RGB traces #{i + 1}", (invalid_rgb, fs))
        for i, invalid_rgb in enumerate(invalid_rgb_traces)
    ]

    run_invalid_input_tests(extract_pos, test_cases)


# ==========================================
# POS SAMPLING FREQUENCY TESTING
# ==========================================

def pos_freq_testing():

    test_cases = [
        (f"Invalid fs={invalid_fs}", (rgb_traces, invalid_fs))
        for invalid_fs in invalid_fs_values
    ]

    run_invalid_input_tests(extract_pos, test_cases)


# ==========================================
# POS WINDOW DURATION TESTING
# ==========================================

def pos_window_duration_testing():

    test_cases = [
        (f"Invalid win_sec={invalid_win}", (rgb_traces, fs, invalid_win))
        for invalid_win in invalid_win_sec
    ]

    run_invalid_input_tests(extract_pos, test_cases)


# ==========================================
# POS MINIMUM WINDOW LENGTH TESTING
# ==========================================

def pos_window_length_testing():

    test_cases = [
        (f"Short window={short_win}", (rgb_traces, fs, short_win))
        for short_win in short_win_values
    ]

    run_invalid_input_tests(extract_pos, test_cases)


# ==========================================
# POS VIDEO LENGTH TESTING
# ==========================================

def pos_video_length_testing():

    test_cases = [
        ("Video shorter than POS window", (short_rgb_traces, fs, 1.6))
    ]

    run_invalid_input_tests(extract_pos, test_cases)


# ==========================================
# POS RGB MEAN VALIDATION TESTING
# ==========================================
def pos_zero_mean_testing():

    test_cases = [
        ("RGB channel mean is zero", (zero_mean_rgb_traces, fs, 1.6))
    ]

    run_invalid_input_tests(extract_pos, test_cases)


# ==========================================
# SYNTHETIC PULSE GENERATION
# ==========================================

def generate_synthetic_pulse(bpm, noise_level=0.0, duration = 10):

    t = np.arange(int(fs * duration)) / fs

    frequency = bpm / 60.0

    pulse = np.sin(2 * np.pi * frequency * t)

    noise = np.random.normal(0, noise_level, len(pulse))

    pulse += noise

    return pulse


# ==========================================
# FFT SIGNAL DURATION TESTING
# ==========================================

def fft_duration_testing():
    test_bpm = 83
    durations = [3, 5, 10, 20]

    for duration in durations:
        pulse = generate_synthetic_pulse(
            test_bpm,
            noise_level=0.1,
            duration=duration
        )

        estimated_bpm = estimate_bpm_fft(pulse, fs)

        error = abs(test_bpm - estimated_bpm)

        print(
            f"Duration: {duration}s | "
            f"Actual: {test_bpm} BPM | "
            f"Estimated: {estimated_bpm:.2f} BPM | "
            f"Error: {error:.2f} BPM"
        )


# ==========================================
# FFT ACCURACY AND NOISE TESTING
# ==========================================

def fft_accuracy_testing():
    test_bpms = [60, 72, 90, 120, 150]
    different_bpms_test = [67, 83, 97, 113, 137]
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5]

    for noise_level in noise_levels:
            print(f"\n--- Noise level: {noise_level} ---")

            for actual_bpm in different_bpms_test:
                pulse = generate_synthetic_pulse(
                    actual_bpm,
                    noise_level
                )

                estimated_bpm = estimate_bpm_fft(pulse, fs)

                error = abs(actual_bpm - estimated_bpm)

                print(
                    f"Actual: {actual_bpm} BPM | "
                    f"Estimated: {estimated_bpm:.2f} BPM | "
                    f"Error: {error:.2f} BPM"
                )


# ==========================================
# FFT COMPETING FREQUENCIES TESTING
# ==========================================

def fft_competing_frequencies_testing():
    actual_bpm = 83
    interference_bpm = 120

    t = np.arange(int(fs * duration)) / fs

    heartbeat = np.sin(2 * np.pi * (actual_bpm / 60) * t)

    interference = 0.3 * np.sin(
        2 * np.pi * (interference_bpm / 60) * t
    )

    pulse = heartbeat + interference

    estimated_bpm = estimate_bpm_fft(pulse, fs)

    error = abs(actual_bpm - estimated_bpm)

    print(
        f"Actual: {actual_bpm} BPM | "
        f"Estimated: {estimated_bpm:.2f} BPM | "
        f"Error: {error:.2f} BPM"
    )

# ==========================================
# TEST RUNNER
# ==========================================

def run_all_failsafe_tests():

    print("\n=== SIGNAL VALIDATION TESTS ===")
    signal_testing()

    print("\n=== INVALID SAMPLING FREQUENCY TESTS ===")
    sample_testing()

    print("\n=== GREEN VALIDATION TESTS ===")
    green_testing()

    print("\n=== GREEN SAMPLING FREQUENCY TESTS ===")
    green_sample_testing()

    print("\n=== CHROM RGB VALIDATION TESTS ===")
    chrom_testing()

    print("\n=== CHROM SAMPLING FREQUENCY TESTS ===")
    chrom_freq_testing()

    print("\n=== CHROM WINDOW DURATION TESTS ===")
    chrom_window_duration_testing()

    print("\n=== CHROM WINDOW LENGTH TESTS ===")
    chrom_window_length_testing()

    print("\n=== CHROM VIDEO LENGTH TESTS ===")
    chrom_video_length_testing()

    print("\n=== POS RGB VALIDATION TESTS ===")
    pos_testing()

    print("\n=== POS SAMPLING FREQUENCY TESTS ===")
    pos_freq_testing()

    print("\n=== POS WINDOW DURATION TESTS ===")
    pos_window_duration_testing()

    print("\n=== POS WINDOW LENGTH TESTS ===")
    pos_window_length_testing()

    print("\n=== POS VIDEO LENGTH TESTS ===")
    pos_video_length_testing()

    print("\n=== FFT PULSE TEST ===")
    fft_accuracy_testing()

    print("\n=== FFT DURATION TEST ===")
    fft_duration_testing()

    print("\n=== FFT COMPLEX TEST ===")
    fft_competing_frequencies_testing()
if __name__ == "__main__":
    run_all_failsafe_tests()
