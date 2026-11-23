# Project TODO

## Milestone 1: rPPG Pulse Extraction & Validation
- [ ] Download UBFC-rPPG dataset into `data/ubfc_rppg/`
- [ ] Track facial landmarks and extract skin ROIs (forehead, left cheek, right cheek)
- [ ] Implement and verify rPPG algorithms:
  - [ ] Raw Green channel
  - [ ] CHROM
  - [ ] POS
- [ ] Apply 4th-order zero-phase Butterworth bandpass filter (0.7–3.0 Hz / 42–180 BPM)
- [ ] Compute Heart Rate via FFT peak ($BPM = f_{peak} \times 60$)
- [ ] Compute MAE against ground truth pulse-oximeter data
- [ ] Pass quantitative gate: MAE $\le 5.0$ BPM on UBFC-rPPG (and beat Green baseline)

## Milestone 2: Baseline Deepfake Detector
- [ ] Download FaceForensics++ c23 clips into `data/faceforensics_c23/`
- [ ] Implement dual-identity connected-component cross-validation split (no identity leakage)
- [ ] Compute pairwise Welch Magnitude-Squared Coherence across ROIs:
  - [ ] Cardiac band (0.7–3.0 Hz)
  - [ ] Residual band (3.0–14.0 Hz)
- [ ] Train classifier (Logistic Regression / SVM) on coherence features
- [ ] Implement video-level majority voting aggregation
- [ ] Pass Go/No-Go gate: Video ROC-AUC $\ge 0.65$
- [ ] (If AUC $< 0.65$): Add pivot features (log-PSD ratios, spectral entropy, PAPR)

## Milestone 3: Adversarial Pulse Injection (Red-Team)
- [ ] Select true-positive fake videos from Milestone 2
- [ ] Implement synthetic pulse generators:
  - [ ] Variant A: Rigid Sine (constant BPM)
  - [ ] Variant B: HRV-Modulated drift
  - [ ] Variant C: Per-patch phase and amplitude jitter
- [ ] Implement feathered skin alpha matte (exclude eyes, mouth)
- [ ] Implement 8-bit TPDF dithering to preserve sub-LSB signals
- [ ] Export attacked videos losslessly (FFV1 rgb24 or PNG sequences)
- [ ] Re-test attacked videos on Milestone 2 detector and calculate attack success rate

## Milestone 4: Corneal Reflection Study & Stress Sweeps
- [ ] Run corneal specular reflection feasibility study on ~20 matched clips:
  - [ ] Filter open eyes via Eye Aspect Ratio (EAR)
  - [ ] Segment specular highlights
  - [ ] Compute bilateral IoU and Normalized Cross-Correlation (NCC)
  - [ ] Document positive or negative feasibility findings
- [ ] Run H.264 compression bitrate sweep (4000 to 250 kbps) and plot ROC-AUC degradation
- [ ] Run window duration comparison (10s vs 30s)
- [ ] Run 1,000-iteration subject bootstrapping for 95% confidence intervals
- [ ] Verify live webcam heart rate demo
