# CardioBreak

Coursework project starter skeleton for remote photoplethysmography (rPPG) video forensics.

## Project Structure

```
CardioBreak/
├── data/               # Place UBFC-rPPG and FaceForensics++ datasets here
├── m1_rppg.py          # Milestone 1: Pulse extraction (Green, CHROM, POS)
├── m2_detector.py      # Milestone 2: Baseline coherence deepfake detector
├── m3_attack.py        # Milestone 3: Adversarial pulse injection
├── m4_eval.py          # Milestone 4: Corneal reflection study & sweeps
├── demo.py             # Live webcam heart rate demo
├── requirements.txt    # Python dependencies
└── TODO.md             # Project task checklist
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run milestone scripts
python m1_rppg.py
python m2_detector.py
python m3_attack.py
python m4_eval.py

# Live webcam demo
python demo.py
```
