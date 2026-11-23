# FaceForensics++ c23 Dataset Directory

Place FaceForensics++ c23 video clips here following the matched-source design:
```
data/faceforensics_c23/
  ├── original_sequences/
  │   └── youtube/
  │       └── c23/
  │           └── videos/
  │               ├── 000.mp4
  │               ├── 001.mp4
  │               └── ...
  └── manipulated_sequences/
      ├── Face2Face/
      │   └── c23/
      │       └── videos/
      │           ├── 000_003.mp4  (source: 000, target: 003)
      │           └── ...
      ├── NeuralTextures/
      │   └── c23/
      │       └── videos/
      ├── FaceSwap/
      │   └── c23/
      │       └── videos/
      └── Deepfakes/
          └── c23/
              └── videos/
```

### Dual-Identity Connected Component Splitting
Remember: Every manipulated video connects a `source_id` and a `target_id`.
To prevent data leakage across folds, clips are partitioned such that entire connected components
of identities are assigned to either train or test folds (via `cardiobreak.detector.graph_split`).
UBFC-rPPG is strictly NOT mixed into this detection experiment to avoid domain shift confounds.
