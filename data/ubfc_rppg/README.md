# UBFC-rPPG Dataset Directory

Place the UBFC-rPPG dataset subjects in this folder according to the structure:
```
data/ubfc_rppg/
  ├── subject1/
  │   ├── vid.avi
  │   └── ground_truth.txt
  ├── subject2/
  │   ├── vid.avi
  │   └── ground_truth.txt
  └── ...
```

The `ground_truth.txt` contains synchronized pulse oximeter readings:
- Line 1: Trace / Photoplethysmogram signal
- Line 2: Ground-truth Heart Rate (BPM)
- Line 3: Timestamp (seconds)

See `scripts/run_m1_rppg_validation.py` for running the validation pipeline.
