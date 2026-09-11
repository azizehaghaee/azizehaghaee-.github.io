# Final Metrics

## Multi-seed validation

| Model | Params | Validation CER (%) |
|---|---:|---:|
| Corrected SVTR Baseline | 4,357,540 | 10.0536 ± 0.2835 |
| No-Contrast Auxiliary | 4,687,653 | 9.6762 ± 0.2424 |
| Signed-Contrast Auxiliary (recovered final set) | 4,687,653 | 9.6933 ± 0.2801 |

Historical Signed validation before checkpoint recovery was 9.6616 ± 0.2184%. After recovering the missing Seed-42 checkpoint from a fresh rerun and completing Seed-2026, the frozen checkpoint set used for final test had validation CER 9.6933 ± 0.2801%.

## Final Signed-Contrast test

| Seed | Best Epoch | Validation CER | Test CER | Test WER | Line Acc. | 1-NED | CTC Loss | beta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 40 | 10.0159 | 11.9041 | 42.6753 | 11.0891 | 87.9603 | 1.0162 | 0.488499 |
| 123 | 38 | 9.5512 | 11.5579 | 41.2338 | 14.4554 | 88.3773 | 1.0162 | 0.491962 |
| 2026 | 38 | 9.5127 | 11.4823 | 40.9481 | 14.4554 | 88.2629 | 1.0055 | 0.458061 |

Mean ± sample SD:

- CER: **11.6481 ± 0.2249%**
- WER: **41.6190 ± 0.9259%**
- Line Accuracy: **13.3333 ± 1.9436%**
- 1-NED: **88.2002 ± 0.2155%**
- CTC Loss: **1.0126 ± 0.0062**
- beta: **0.479507 ± 0.018653**

## Matched Seed-42 final-test comparison

| Model | Test CER | Test WER | Line Acc. | 1-NED |
|---|---:|---:|---:|---:|
| Corrected SVTR Baseline | 12.5576 | 44.6883 | 9.9010 | 87.5092 |
| Dot-Aware (absolute local contrast) | 12.1894 | 43.2597 | 11.0891 | 87.8309 |
| Signed-Contrast | **11.9041** | **42.6753** | **11.0891** | **87.9603** |

Signed vs Baseline Seed-42: -0.6535 percentage points CER, approximately 5.20% relative CER reduction.
