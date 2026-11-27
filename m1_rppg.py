"""Milestone 1: rPPG Pulse Extraction & Validation (UBFC-rPPG).

Starter skeleton for implementing:
- 4th-order zero-phase Butterworth bandpass filter (0.7-3.0 Hz)
- Green, CHROM, and POS extraction algorithms
- FFT spectral peak BPM estimation
- Mean Absolute Error (MAE) calculation against ground truth
"""

""" -------------------IMPORTANT NOTE-------------------
CardioBreak requires the MediaPipe Face Landmarker model.

Download:
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Place it at:

models/face_landmarker.task

"""

import numpy as np
from scipy import signal
import cv2
import mediapipe as mp


# ==========================================
# GLOBAL LANDMARKS & VARIABLES
# ==========================================

face_oval_indices = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]

eye_indices = [33, 133, 159, 145,
            362, 263, 386, 374]

mouth_indices = [
    61, 291,
    13, 14
]


forehead_boundaries = {
    "upper": [103, 67, 109, 10, 338, 297, 332],
    "lower": [71, 68, 104, 69, 108, 337, 299, 333]
}


UPPER_CHEEK_OFFSET_RATIO = 0.36

INNER_CHEEK_HORIZONTAL_RATIO = 0.12

INNER_CHEEK_VERTICAL_RATIO = 0.08

# ==========================================
# SETUP HELPER FUNCTIONS
# ==========================================

def validate_rgb_traces(rgb_traces, fs):
    rgb_traces = np.asarray(rgb_traces, dtype=np.float64)
    if rgb_traces.ndim != 2 or rgb_traces.shape[1] != 3:
        raise ValueError(
            "RGB traces must have shape (N, 3)"
        )

    if rgb_traces.shape[0] == 0:
        raise ValueError(
            "RGB traces cannot be empty"
            )

    if not np.all(np.isfinite(rgb_traces)):
        raise ValueError(
            "RGB traces cannot contain NaN or infinite values"
            )

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(
            "Sampling Frequency must be a positive finite number"
        )

    return rgb_traces

def padlen_calculation(sos):
    pad_length = 3 * (
        2 * len(sos) + 1
        - min (
            (sos[:,2] == 0).sum(),
            (sos[:, 5] == 0).sum()
        )
    )
    return pad_length

def signal_failsafe(sig):
    sig = np.asarray(sig, dtype = np.float64)

    if sig.ndim != 1:
        raise ValueError("Signal must be a 1D array")

    if len(sig) == 0:
        raise ValueError("Signal cannot be empty")

    if not np.all(np.isfinite(sig)):
        raise ValueError("Signal cannot contain NaN or infinite values")

    return sig

# ==========================================
# SETUP :
# DETREND & BANDPASS FILTER
# ==========================================

def filter_rppg(sig: np.ndarray, fs: float = 30.0, low_hz: float = 0.7, high_hz: float = 3.0) -> np.ndarray:

    if fs <= 0 or not np.isfinite(fs):
        raise ValueError("Sampling frequency must be positive")

    if not (0 < low_hz < high_hz < fs / 2):
        raise ValueError("Invalid frequency range for the given sampling frequency")

    sig = signal_failsafe(sig)
    sig = signal.detrend(sig, type = "linear")

    sos = signal.butter(4, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")

    padlen = padlen_calculation(sos)


    if len(sig) <= padlen:
        raise ValueError(
            "Signal is too short for zero-phase filtering"
        )

    filtered = signal.sosfiltfilt(sos, sig)
    return filtered

# ==========================================
# GREEN EXTRACTION
# ==========================================

def extract_green(rgb_traces: np.ndarray, fs: float = 30.0) -> np.ndarray:
    rgb_traces = validate_rgb_traces(rgb_traces, fs)

    green = rgb_traces [:, 1]
    pulse = filter_rppg(green, fs)

    return pulse

# ==========================================
# CHROM EXTRACTION
# ==========================================

def extract_chrom(rgb_traces: np.ndarray, fs: float = 30.0, win_sec: float = 1.6) -> np.ndarray:

    rgb_traces = validate_rgb_traces(rgb_traces, fs)

    if not np.isfinite(win_sec) or win_sec <= 0:
        raise ValueError("Window duration must be positive and finite")

    win_samples = int(round(win_sec * fs))
    if len(rgb_traces) < win_samples:
        raise ValueError("Video is too short for a CHROM window")

    step = win_samples // 2
    if step <= 0:
        raise ValueError("Step must be a positive number")

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

        Xs = signal.detrend(Xs, type="linear")
        Ys = signal.detrend(Ys, type="linear")

        if np.std(Ys) < 1e-8:
            alpha = 0.0
        else:
            alpha = np.std(Xs) / np.std(Ys)

        window_pulse = Xs - alpha * Ys

        window_pulse -= np.mean(window_pulse)

        pulse_sum[start:end] += window_pulse
        weight_sum[start:end] += 1

    if np.any(weight_sum == 0):
        raise ValueError("Some frames were not covered by CHROM windows")

    pulse = pulse_sum / weight_sum
    pulse = filter_rppg(pulse, fs)
    return pulse

# ==========================================
# POS EXTRACTION
# ==========================================

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

# ==========================================
# BPM ESTIMATE
# ==========================================

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

# ==========================================
# FACE RECOGNITION
# ==========================================

# ==========================================
# FACE LANDMARKER INITIALIZATION
# ==========================================

def create_face_landmarker(model_path):

    """
    Initialize MediaPipe Face Landmarker.

    Args:
        model_path: Path to the .task model file.

    Returns:
        Initialized Face Landmarker object.
    """

    # Configure the location of the face landmark model.
    base_options = mp.tasks.BaseOptions(
        model_asset_path = model_path
    )

    # Configure the detector's operating settings.
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options = base_options,
        running_mode = mp.tasks.vision.RunningMode.VIDEO,
        num_faces = 1
    )

    # Create the Face Landmarker using the configuration.
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
        options
    )

    return landmarker

# ==========================================
# LANDMARK COORDINATE CONVERSION
# ==========================================

def get_landmark_points(landmarks, indices, width, height):

    """
    Convert selected normalized facial landmarks into pixel coordinates.

    Args:
        landmarks: Collection of facial landmarks.
        indices: Indices of the landmarks we want to retrieve.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        NumPy array containing [x, y] pixel coordinates.
    """

    points = []

    # Retrieve each requested landmark and convert its coordinates.
    for index in indices:
        landmark = landmarks[index]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append([x, y])

    # Convert the coordinate list into an integer NumPy array.
    return np.array(points, dtype = np.int32)


# ==========================================
# FACE DETECTION AND FOREHEAD ROI TESTING
# ==========================================

def test_face_detection(video_path, model_path):

    """
    Test facial landmark detection and forehead ROI construction.

    Steps:
        1. Open the video and read its first frame.
        2. Initialize the MediaPipe Face Landmarker.
        3. Convert the frame into the format required by MediaPipe.
        4. Detect facial landmarks.
        5. Extract the landmarks defining the forehead boundaries.
        6. Construct a forehead polygon and generate its binary mask.
        7. Visualize the forehead ROI to verify its accuracy.
        8. Release the MediaPipe and video capture resources.
    """

    # VIDEO INITIALIZATION
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    ret, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_index = 0

    if not ret:
        cap.release()
        raise RuntimeError("Could not read first frame")

    # MEDIAPIPE INITIALIZATION
    landmarker = create_face_landmarker(model_path)

    # FRAME PREPARATION
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp_ms = calculate_timestamp(frame_index, fps)

    # FACE DETECTION
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]
        height, width, _ = frame.shape

        # ==========================================
        # FOREHEAD ROI CONSTRUCTION && FOREHEAD MASK
        # ==========================================

        # Construct the forehead ROI mask
        forehead_roi_mask = create_roi_mask(
            landmarks,
            forehead_boundaries,
            width,
            height
        )

        # Visualize the forehead ROI
        overlay = frame.copy()

        overlay[forehead_roi_mask == 255] = (
            overlay[forehead_roi_mask == 255] * 0.5
            + np.array([0, 255, 0]) * 0.5
        ).astype(np.uint8)

        cv2.imshow("Forehead ROI Overlay", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # ==========================================
        # FACE BOUNDARY SETTING
        # ==========================================

        # Extract eyes landmarks to establish the lower cheek reference
        eye_points = get_landmark_points(
            landmarks,
            eye_indices,
            width,
            height
        )


        # Extract mouth landmarks to establish the lower cheek reference
        mouth_points = get_landmark_points(
            landmarks,
            mouth_indices,
            width,
            height
        )

        # Calculate the dynamic lower cheek boundary
        lower_cheek_points = calculate_lower_cheek_boundary(
            eye_points,
            mouth_points
        )



        # ==========================================
        # FIRST & SECOND CHEEK CONSTRUCTION
        # ==========================================

        # Construct the first cheek ROI
        first_cheek_mask = create_first_cheek_roi_mask(
            landmarks,
            lower_cheek_points[1],
            face_oval_indices,
            width,
            height
        )

        # Construct the second cheek ROI
        second_cheek_mask = create_second_cheek_roi_mask(
            landmarks,
            lower_cheek_points[0],
            face_oval_indices,
            width,
            height
        )


        # ==========================================
        # FIRST & SECOND CHEEK VISUALIZATION
        # ==========================================

        # First Cheek
        overlay = frame.copy()

        overlay[first_cheek_mask == 255] = (
            overlay[first_cheek_mask == 255] * 0.5
            + np.array([0, 255, 0]) * 0.5
        ).astype(np.uint8)

        cv2.imshow("First Cheek ROI", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        overlay = frame.copy()

        # Second Cheek
        overlay[second_cheek_mask == 255] = (
            overlay[second_cheek_mask == 255] * 0.5
            + np.array([0, 255, 0]) * 0.5
        ).astype(np.uint8)

        cv2.imshow("Second Cheek ROI", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # ==========================================
        # SYNCRONIZED VISUALIZATION
        # ==========================================

        # Combine both cheek masks
        combined_cheek_mask = cv2.bitwise_or(
            first_cheek_mask,
            second_cheek_mask
        )

        overlay = frame.copy()

        overlay[combined_cheek_mask == 255] = (
            overlay[combined_cheek_mask == 255] * 0.5
            + np.array([0, 255, 0]) * 0.5
        ).astype(np.uint8)

        # Display the final ROI visualization
        cv2.imshow("Cheek ROI", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("Landmarks detected:", len(landmarks))

    else:
        print("No face detected")

    # Release video capture and MediaPipe resources
    landmarker.close()
    cap.release()

# ==========================================
# ROI HELPER FUNCTIONS
# ==========================================

def calculate_timestamp(frame_index, fps):
    time_stamp = (frame_index / fps) * 1000
    return int(time_stamp)

def calculate_lower_cheek_boundary(eye_points, mouth_points, ratio=0.55):
    lower_eye_points = eye_points[[3, 7]]
    mouth_corner_points = mouth_points[[0, 1]]

    lower_boundary = (
        lower_eye_points
        + ratio * (mouth_corner_points - lower_eye_points)
    )

    return np.round(lower_boundary).astype(np.int32)

def create_first_cheek_roi_mask(
    landmarks,
    lower_anchor,
    face_oval_indices,
    width,
    height
):
    # Extract the landmarks defining the cheek polygon
    cheek_indices = [
        362,  # Inner eye corner
        374,  # Lower eyelid
        263,  # Outer eye corner
        454,  # Outer face contour
        323,  # Lower outer face contour
        327   # Nostril base
    ]

    cheek_points = get_landmark_points(
        landmarks,
        cheek_indices,
        width,
        height
    )

    inner_eye, lower_lid, outer_eye, outer_upper, outer_lower, inner_lower = cheek_points

    # Calculate eye width to scale adjustments with face size
    eye_width = np.linalg.norm(outer_eye - inner_eye)

    # Calculate the downward direction perpendicular to the eye line
    eye_direction = outer_eye - inner_eye
    downward_direction = np.array([
        -eye_direction[1],
        eye_direction[0]
    ])

    direction_norm = np.linalg.norm(downward_direction)

    if direction_norm < 1e-6:
        raise RuntimeError("Invalid downward direction: zero-length vector")

    downward_direction = downward_direction / direction_norm
    if downward_direction[1] < 0:
        downward_direction *= -1

    # Move the upper boundary below the eyeglasses
    upper_offset = eye_width * UPPER_CHEEK_OFFSET_RATIO

    inner_eye = inner_eye + downward_direction * upper_offset
    lower_lid = lower_lid + downward_direction * upper_offset
    outer_eye = outer_eye + downward_direction * upper_offset

    # Move the inner boundary away from the nose
    inner_lower = inner_lower + np.array([
        eye_width * INNER_CHEEK_HORIZONTAL_RATIO,
        -eye_width * INNER_CHEEK_VERTICAL_RATIO
    ])

    # Define the polygon
    cheek_polygon = np.array([
        inner_eye,
        lower_lid,
        outer_eye,
        outer_upper,
        outer_lower,
        lower_anchor,
        inner_lower
    ], dtype=np.int32)

    # Create the cheek mask
    cheek_mask = np.zeros((height, width), dtype=np.uint8)

    cv2.fillPoly(
        cheek_mask,
        [cheek_polygon],
        255
    )

    # Create the face oval mask
    face_oval_points = get_landmark_points(
        landmarks,
        face_oval_indices,
        width,
        height
    )

    face_oval_mask = np.zeros((height, width), dtype=np.uint8)

    cv2.fillPoly(
        face_oval_mask,
        [face_oval_points.astype(np.int32)],
        255
    )

    # Keep only the cheek area inside the face oval
    cheek_mask = cv2.bitwise_and(
        cheek_mask,
        face_oval_mask
    )

    return cheek_mask

def create_second_cheek_roi_mask(
    landmarks,
    lower_anchor,
    face_oval_indices,
    width,
    height
):
    # Extract landmarks for the second cheek
    cheek_indices = [
        133,  # Inner eye corner
        145,  # Lower eyelid
        33,   # Outer eye corner
        234,  # Outer face contour
        93,  # Lower outer face contour
        98    # Nostril base
    ]

    cheek_points = get_landmark_points(
        landmarks,
        cheek_indices,
        width,
        height
    )

    inner_eye, lower_lid, outer_eye, outer_upper, outer_lower, inner_lower = cheek_points

    # Calculate eye width
    eye_width = np.linalg.norm(outer_eye - inner_eye)

    # Calculate the downward direction
    eye_direction = outer_eye - inner_eye
    downward_direction = np.array([
        -eye_direction[1],
        eye_direction[0]
    ])

    direction_norm = np.linalg.norm(downward_direction)

    if direction_norm < 1e-6:
        raise RuntimeError("Invalid downward direction: zero-length vector")

    downward_direction = downward_direction / direction_norm

    if downward_direction[1] < 0:
        downward_direction *= -1

    # Move the upper boundary below the eyeglasses
    upper_offset = eye_width * UPPER_CHEEK_OFFSET_RATIO

    inner_eye = inner_eye + downward_direction * upper_offset
    lower_lid = lower_lid + downward_direction * upper_offset
    outer_eye = outer_eye + downward_direction * upper_offset

    # Move the inner boundary away from the nose
    inner_lower = inner_lower + np.array([
        -eye_width * INNER_CHEEK_HORIZONTAL_RATIO,
        -eye_width * INNER_CHEEK_VERTICAL_RATIO
    ])

    # Define the polygon
    cheek_polygon = np.array([
        inner_eye,
        lower_lid,
        outer_eye,
        outer_upper,
        outer_lower,
        lower_anchor,
        inner_lower
    ], dtype=np.int32)

    # Create the cheek mask
    cheek_mask = np.zeros((height, width), dtype=np.uint8)

    cv2.fillPoly(
        cheek_mask,
        [cheek_polygon],
        255
    )

    # Create the face oval mask
    face_oval_points = get_landmark_points(
        landmarks,
        face_oval_indices,
        width,
        height
    )

    face_oval_mask = np.zeros((height, width), dtype=np.uint8)

    cv2.fillPoly(
        face_oval_mask,
        [face_oval_points.astype(np.int32)],
        255
    )

    # Keep only the cheek area inside the face oval
    cheek_mask = cv2.bitwise_and(
        cheek_mask,
        face_oval_mask
    )

    return cheek_mask

def create_roi_mask(landmarks, boundaries, width, height):

    # Extract upper forehead boundary landmarks
    upper_points = get_landmark_points(
        landmarks,
        boundaries["upper"],
        width,
        height
    )

    # Extract lower forehead boundary landmarks
    lower_points = get_landmark_points(
        landmarks,
        boundaries["lower"],
        width,
        height
    )

    # Combine both boundaries into a closed polygon
    forehead_polygon = np.vstack((
        upper_points,
        lower_points[::-1]
    ))

    # Create an empty mask
    roi_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    # Fill the forehead polygon
    cv2.fillPoly(
        roi_mask,
        [forehead_polygon],
        255
    )

    return roi_mask

# ==========================================
# TESTING FUNCTIONS FOR MAIN RUNS
# ==========================================

#Global rgb_trace for testing
rgb_traces = np.array([
    [120.0, 130.0, 110.0],
    [121.0, 131.0, 111.0],
    [122.0, 132.0, 112.0],
    [123.0, 133.0, 113.0]
])

def signal_testing():
    invalid_signals = [
        np.array([]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([1.0, 2.0, np.nan, 4.0]),
        np.array([1.0, 2.0, np.inf, 4.0])
    ]

    for invalid_sig in invalid_signals:
        try:
            filtered = filter_rppg(invalid_sig, fs)
            return filtered
        except ValueError as error:
            print(f"Caught expected error: {error}")

def sample_testing():
    invalid_fs_values = [0, -30, np.nan, np.inf]

    for invalid_fs in invalid_fs_values:
        try:
            filter_rppg(sig, fs=invalid_fs)
        except ValueError as error:
            print(f"fs={invalid_fs} -> Caught expected error: {error}")

def green_testing():
    invalid_rgb_traces = [
        np.array([]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[1.0, np.nan, 3.0]]),
        np.array([[1.0, np.inf, 3.0]])
    ]

    for invalid_rgb in invalid_rgb_traces:
        try:
            extract_green(invalid_rgb, fs)
        except ValueError as error:
            print("Caught expected error:", error)

def green_sample_testing():

    # Replace invalid_fs with 30 for short signal error
    invalid_fs_values = [0, -30, np.nan, np.inf]

    rgb_traces = np.array([
        [120.0, 130.0, 110.0],
        [121.0, 131.0, 111.0],
        [122.0, 132.0, 112.0],
        [123.0, 133.0, 113.0]
    ])

    for invalid_fs in invalid_fs_values:
        try:
            extract_green(rgb_traces, invalid_fs)
        except ValueError as error:
            print(f"fs={invalid_fs} -> Caught expected error: {error}")

def chrom_testing(fs):
    invalid_rgb_traces = [
        np.array([]),                                  # Empty array
        np.array([[1.0, 2.0], [3.0, 4.0]]),           # Wrong shape
        np.array([[1.0, np.nan, 3.0]]),              # NaN
        np.array([[1.0, np.inf, 3.0]])               # Infinity
    ]

    for invalid_rgb in invalid_rgb_traces:
        try:
            extract_chrom(invalid_rgb, fs)
        except ValueError as error:
            print("Caught expected error:", error)

def chrom_freq_testing():
    invalid_fs_values = [0, -30, np.nan, np.inf]
    rgb_traces = np.array([
        [120.0, 130.0, 110.0],
        [121.0, 131.0, 111.0],
        [122.0, 132.0, 112.0],
        [123.0, 133.0, 113.0]
    ])

    for invalid_fs in invalid_fs_values:
        try:
            extract_chrom(rgb_traces, invalid_fs)
        except ValueError as error:
            print(f"fs={invalid_fs} -> Caught expected error: {error}")

def chrom_window_duration_testing():
    invalid_win_sec = [0, -1.0, np.nan, np.inf]

    for invalid_win in invalid_win_sec:
        try:
            extract_chrom(rgb_traces, fs, invalid_win)
        except ValueError as error:
            print(f"win_sec={invalid_win} -> Caught expected error: {error}")

def chrom_window_length_testing():
    short_win_values = [0.01, 0.1, 0.5]

    for short_win in short_win_values:
        try:
            extract_chrom(rgb_traces, fs, short_win)
        except ValueError as error:
            print(f"win_sec={short_win} -> Caught expected error: {error}")

def chrom_video_length_testing():
    short_rgb_traces = np.random.rand(20, 3) * 255

    try:
        extract_chrom(short_rgb_traces, fs, 1.6)
    except ValueError as error:
        print("Caught expected error:", error)

# ==========================================
# MAIN RUNS
# ==========================================

def run_m1():
    """TODO: Load UBFC-rPPG video and ground truth, compute MAE, and check gate (MAE <= 5.0 BPM)."""
    print("Running Milestone 1...")
    # TODO: Add your validation code here

if __name__ == "__main__":

    video_path= r"data/ubfc_rppg/6-gt/vid.avi"
    model_path = r"models/face_landmarker.task"
    # test_face_detection(video_path, model_path)

    fs = 30.0
    duration = 10

    t = np.arange(0, duration, 1/fs)

    sig = (
        120
        + 2 * np.sin(2 * np.pi * 1.2 * t)
        + 0.5 * np.sin(2 * np.pi * 5 * t)
    )

    # ==============================
    # SIGNAL VALIDATION TESTING
    # ==============================

    # signal_test = signal_testing()

    # ==============================
    # INVALID SAMPLING FREQUENCY TESTS
    # ==============================

    # sample_test = sample_testing()

    # ==============================
    # EXTRACT_GREEN VALIDATION TESTING
    # ==============================

    # green_test = green_testing()

    # ==============================
    # EXTRACT_GREEN SAMPLING FREQUENCY TESTING
    # ==============================

    # green_sample_test = green_sample_testing()

    # ==============================
    # CHROM RGB VALIDATION TESTING
    # ==============================

    # chrom_test = chrom_testing(fs)

    # ==============================
    # CHROM SAMPLING FREQUENCY TESTING
    # ==============================

    # chrom_freq_test = chrom_freq_testing()

    # ==============================
    # CHROM WINDOW DURATION TESTING
    # ==============================

    # chrom_window_test = chrom_window_testing()

    # ==============================
    # CHROM WINDOW LENGTH TESTING
    # ==============================

    # chrom_window_length_test = chrom_window_length_testing()

    # ==============================
    # CHROM VIDEO LENGTH TESTING
    # ==============================

    chrom_video_length_test = chrom_video_length_testing()

    # run_m1()
