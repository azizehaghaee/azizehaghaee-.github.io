# Persian Handwritten Text Recognition Research

This directory collects the code, experiment protocol, and final reported metrics from the Persian handwritten text-line recognition work carried out on the PHD line dataset.

## Final dataset / protocol

- 5,014 text-line images from 500 writers
- writer-independent split: Train 4,025 / Validation 484 / Test 505
- charset: 67 symbols; CTC classes: 68 including blank
- input: RGB 64x768 with aspect-ratio-preserving resize + white padding
- no image mirroring and no ground-truth reversal
- Persian RTL is handled by reversing the temporal logits before CTC
- temporal length: 192; P99 label length: 123; maximum label length: 163

## Repository layout

- `configs/final_protocol.yaml` — frozen experimental protocol
- `svtr/final_test_signed_3seed.py` — exact final three-seed Signed-Contrast test cell used for final reporting
- `svtr/signed_checkpoint_recovery.py` — exact recovery/retraining logic for the missing Signed checkpoints
- `svtr/experiment_stages.md` — Baseline, Dot-Aware, No-Contrast, Signed and multi-seed chronology/results
- `results/final_metrics.md` — article-ready final metrics
- `legacy/` — earlier source files preserved as historical material

## Reproducibility note

The final-test and recovery scripts are preserved from the final Kaggle workflow and intentionally expect the notebook objects used in that workflow (`SVTRTinyBackbone`, datasets, `svtr_collate_fn`, and `charset`) to already be defined. Earlier historical experiments were often executed as notebook cells rather than retained as standalone `.py` files. Their validated settings and results are documented in `svtr/experiment_stages.md`; code is not silently invented where an original standalone source was not retained.

## Main final SVTR result

The frozen Signed-Contrast model (4,687,653 parameters), selected using validation only, achieved across three final test seeds:

- CER: **11.6481 ± 0.2249%**
- WER: **41.6190 ± 0.9259%**
- Line Accuracy: **13.3333 ± 1.9436%**
- 1-NED: **88.2002 ± 0.2155%**

For the matched Seed-42 comparison, baseline CER was 12.5576% and Signed-Contrast CER was 11.9041%, a 0.6535 percentage-point absolute reduction (about 5.20% relative CER reduction).

## Scientific interpretation

The multi-seed ablation supports the contribution of the Stage-2 auxiliary path and gated residual fusion more strongly than a consistent advantage of one particular local-contrast operator. No-Contrast and Signed-Contrast were very close on validation, so the gain should not be attributed solely to the Signed local-contrast operator.
