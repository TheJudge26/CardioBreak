# Lossless Adversarial Injected Samples (Milestone 3)

This directory stores adversarial attack samples injected with synthetic pulse modulations.

### Signal Integrity Rules
1. **Lossless FFV1 (rgb24) or PNG sequences only**: Never re-encode injected videos with H.264 or yuv420p.
   Re-encoding confounds pulse spoofing with compression artifacts.
2. **8-bit TPDF Dithering applied**: Triangular Probability Density Function noise preserves sub-LSB signals
   and prevents artificial quantization harmonics.
3. Attack variants:
   - **Variant A**: Rigid Sine ($f_0 \in [1.0, 1.67]$ Hz)
   - **Variant B**: HRV-Modulated ($\Delta f_{HRV}$)
   - **Variant C**: Adaptive Per-Patch Phase & Amplitude Jitter ($\delta \phi_k, a_k$)

Intermediate lossless files should be archived or cleared once feature extraction is complete to stay within the laptop compute/storage budget (< 15 GB).
