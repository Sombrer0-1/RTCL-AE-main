# 2026-09-22 baseline archive

This is a small feasibility subset of the RTSS paper experiments, not a full reproduction or a platform-matched comparison. Original CSV values and all six raw logs are preserved unchanged, including earlier attempts. Root-level AE tables predate these local experiments.

SplitCIFAR10, ResNet-20, replay, 50,000 training samples. `results.csv` reports historical wall times and QPS (50,000 / wall time). The exact historical launch commands and seeds were not recorded. AOCL logs show an initial training batch of 64; this is forced by the original entry point. Do not interpret the speedup as scheduler-only improvement at matched batch size.

Historical results: sequential 215.95 s / accuracy 0.4446; fully parallel 44.73 s / 0.4484; AOCL 41.02 s / 0.3487; AOCL gamma=eta=0.1 42.12 s / 0.3418. Accuracy and throughput must be assessed jointly. Runtime prints full-test-stream snapshot accuracy, not a strict prequential streaming measurement.

The archived eval-worker change ensures a final post-training evaluation is attempted even when fast training finishes while evaluation is sleeping/stopped. Side artifacts are retained as found, without claiming a one-to-one mapping to runs.

Local dependency: Avalanche eb075be393e1f458b2c352514ff6c17b5a2c0f4e plus repository `modified/` overlays; dependency checkout is excluded from git. Current host has RTX 5090 GPUs; current Python environment uses PyTorch 2.10.0+cu128. Original experiments have no complete environment manifest.
