# ==================================================================================================
# MULTI-SEED VALIDATION EXPERIMENT
# NO-CONTRAST vs SIGNED-CONTRAST
#
# Seeds:
#   existing reference : 42
#   new runs           : 123, 2026
#
# Models:
#   A) No-Contrast      : Z = X
#   B) Signed-Contrast  : Z = X - AvgPool3x3(X)
#
# IMPORTANT:
#   ✓ EXACT SAME parameterized architecture
#   ✓ EXACT SAME number of parameters = 4,687,653
#   ✓ Same training protocol
#   ✓ Same seed used for paired models
#   ✓ Same writer-independent Train/Val split
#   ✓ TEST IS NEVER EVALUATED
#   ✓ Best checkpoint selected by Validation CER
#   ✓ Emergency checkpoint every 100 batches
#   ✓ Robust RNG restore
#   ✓ Completed runs are automatically skipped
#   ✓ Interrupted runs are automatically resumed
#
# FINAL OUTPUT:
#   - Per-seed table
#   - Mean ± Sample Std across seeds 42,123,2026
#   - Paired Signed - NoContrast CER differences
# ==================================================================================================
