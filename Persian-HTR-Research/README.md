# Persian Handwritten Text Recognition Research

This directory is an **archival code repository** for the Persian handwritten text-line recognition research. Its purpose is to preserve the implementation, experiment protocol, and final reported metrics so the work can be revisited in future years.

> **Dataset policy:** the dataset itself is intentionally **not included** in this repository. Only code, experiment settings, and result documentation are preserved.

## Final dataset / protocol reference

- 5,014 text-line images from 500 writers
- writer-independent split: Train 4,025 / Validation 484 / Test 505
- charset: 67 symbols; CTC classes: 68 including blank
- input: RGB 64x768 with aspect-ratio-preserving resize + white padding
- no image mirroring and no ground-truth reversal
- Persian RTL is handled by reversing the temporal logits before CTC
- temporal length: 192; P99 label length: 123; maximum label length: 163

## Repository layout

- `configs/final_protocol.yaml` — frozen experimental protocol
- `svtr/train_all_variants.py` — consolidated Baseline / Dot-Aware / No-Contrast / Signed training code
- `svtr/final_test_signed_3seed.py` — final three-seed Signed-Contrast test code used for final reporting
- `svtr/signed_checkpoint_recovery.py` — checkpoint recovery/retraining logic used before final test
- `svtr/experiment_stages.md` — Baseline, Dot-Aware, No-Contrast, Signed and multi-seed chronology/results
- `results/final_metrics.md` — article-ready final metrics
- `legacy/` — earlier source files preserved as historical material

## Reproducibility / archival note

The purpose of this repository is **long-term code preservation**, not dataset distribution. Some historical experiments were executed directly as Kaggle notebook cells and were not retained as standalone source files. Where original code was available, it is preserved. Where a standalone historical source was not retained, the validated configuration and experimental result are documented rather than silently reconstructing an unverified "original" file.

The final-test and recovery scripts expect the notebook objects used in the final Kaggle workflow (`SVTRTinyBackbone`, datasets, `svtr_collate_fn`, and `charset`) to already be defined. Dataset paths therefore need to be supplied by the future user when the code is run again.

## Main final SVTR result

The frozen Signed-Contrast model (4,687,653 parameters), selected using validation only, achieved across three final test seeds:

- CER: **11.6481 ± 0.2249%**
- WER: **41.6190 ± 0.9259%**
- Line Accuracy: **13.3333 ± 1.9436%**
- 1-NED: **88.2002 ± 0.2155%**

For the matched Seed-42 comparison, baseline CER was 12.5576% and Signed-Contrast CER was 11.9041%, a 0.6535 percentage-point absolute reduction (about 5.20% relative CER reduction).

## Scientific interpretation

The multi-seed ablation supports the contribution of the Stage-2 auxiliary path and gated residual fusion more strongly than a consistent advantage of one particular local-contrast operator. No-Contrast and Signed-Contrast were very close on validation, so the gain should not be attributed solely to the Signed local-contrast operator.
