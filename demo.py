"""Live Real-Time Webcam rPPG Extractor (Section 7 Live Demo).

Starter skeleton for running an interactive webcam heart rate demo:
- Captures frames from webcam (cv2.VideoCapture)
- Localizes face & forehead ROI
- Computes real-time rPPG pulse (POS or CHROM)
- Displays estimated BPM overlay on video feed
"""

import cv2
import numpy as np


def run_webcam_demo(camera_id: int = 0):
    """Run interactive webcam demo.
    
    TODO:
    1. Open cv2.VideoCapture(camera_id).
    2. In a loop, extract forehead skin RGB mean per frame.
    3. Maintain sliding window of ~10 seconds.
    4. Run POS/CHROM extraction and FFT peak estimation every 0.5s.
    5. Render BPM readout on screen using cv2.putText.
    6. Exit when 'q' is pressed.
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Could not open camera {camera_id}")
        return

    print("Live demo skeleton. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # TODO: Add your live face tracking and BPM overlay here
        cv2.putText(frame, "CardioBreak Demo (Skeleton)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("CardioBreak", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_demo()
