# SVTR Experimental Stages

## 1. Corrected baseline

Final corrected baseline architecture includes the missing final LayerNorm(256) after Stage 3. Parameter count: 4,357,540.

Three-seed validation:

- Seed 42: CER 10.3805%, WER 40.0191%, LineAcc 13.6364%, 1-NED 89.5022%, best epoch 40
- Seed 123: CER 9.9055%, WER 38.0530%, LineAcc 16.9421%, 1-NED 90.0910%, best epoch 39
- Seed 2026: CER 9.8747%, WER 38.0939%, LineAcc 15.2893%, 1-NED 89.9172%, best epoch 39
- Mean CER: 10.0536 ± 0.2835%

Seed-42 final Test: CER 12.5576%, WER 44.6883%, LineAcc 9.9010%, 1-NED 87.5092%.

## 2. Dot-Aware absolute local contrast

At Stage 2, use `abs(X - AvgPool3x3(X))`, then 1x1 projection, depthwise 3x3, pointwise 1x1, and 3x3 stride (2,1) to align with Stage 3. Gated residual fusion is applied before the corrected final LayerNorm.

Parameters: 4,687,653. Seed-42 Validation CER 10.1751%. Seed-42 Test CER 12.1894%.

## 3. Parameter-matched No-Contrast control

Identical auxiliary branch, parameters, optimizer, fusion and final LayerNorm, but branch input is `X` instead of a local-contrast response.

Three-seed validation CER: 9.6762 ± 0.2424%.

## 4. Signed-Contrast

Identical auxiliary branch with `X - AvgPool3x3(X)` (no absolute value).

Recovered frozen validation checkpoint set used for the final Test:

- Seed 42: 10.0159%, best epoch 40
- Seed 123: 9.5512%, best epoch 38
- Seed 2026: 9.5127%, best epoch 38
- Mean CER: 9.6933 ± 0.2801%

Final three-seed Test: CER 11.6481 ± 0.2249%, WER 41.6190 ± 0.9259%.

## 5. Interpretation

The equal-parameter No-Contrast control performs essentially the same as Signed-Contrast. Therefore, the strongest supported claim is that the Stage-2 auxiliary branch plus gated residual fusion improves the corrected baseline; the gain cannot be assigned uniquely to the signed local-contrast operator.

## 6. Test policy

Architecture selection was performed on Validation only. The final Signed checkpoints were frozen before final Test evaluation. Test results were not used for architecture or hyperparameter selection.
