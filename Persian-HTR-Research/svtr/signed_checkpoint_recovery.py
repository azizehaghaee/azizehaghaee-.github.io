"""
Signed checkpoint recovery helper used before final Test.

Purpose of the final recovery run:
- Seed 42: original best checkpoint was unavailable, so retrain Signed-Contrast from scratch.
- Seed 123: preserve existing completed best checkpoint.
- Seed 2026: resume the backed-up mid-epoch checkpoint and finish to epoch 40.
- Never evaluate Test in this recovery stage.

This file records the exact frozen paths/results used after recovery. The full training loop is the same
final protocol implemented in `train_all_variants.py`; use that file for a fresh Signed run. For Seed-2026
resume, copy the immutable backup folder before continuing so the source backup is never modified.
"""
from pathlib import Path
import shutil, torch

BACKUP_ROOT=Path("/kaggle/input/datasets/sainasadeghipour/svtr-backup/signed")
RECOVERY_ROOT=Path("/kaggle/working/SVTR_SIGNED_CHECKPOINT_RECOVERY")
RECOVERY_ROOT.mkdir(parents=True,exist_ok=True)

# Preserve existing completed/backed-up seeds without overwriting a recovery already in progress.
for seed in [123,2026]:
    src=BACKUP_ROOT/f"seed_{seed}"; dst=RECOVERY_ROOT/f"seed_{seed}"
    if not src.exists(): raise RuntimeError(f"Missing immutable backup: {src}")
    if not dst.exists(): shutil.copytree(src,dst)

# Seed 123 validation check.
p123=RECOVERY_ROOT/"seed_123/checkpoints/best.pt"; c123=torch.load(p123,map_location="cpu",weights_only=False)
assert c123.get("mode")=="signed" and int(c123.get("seed"))==123 and int(c123.get("parameters"))==4_687_653
assert int(c123.get("best_epoch"))==38 and abs(float(c123.get("best_val_cer"))*100-9.5512)<0.001
print("Seed123 preserved:",p123)

# Seed 2026 must resume from the copied checkpoint, not start fresh.
r2026=RECOVERY_ROOT/"seed_2026/checkpoints/resume.pt"; c2026=torch.load(r2026,map_location="cpu",weights_only=False)
assert c2026.get("mode")=="signed" and int(c2026.get("seed"))==2026
print("Seed2026 resume checkpoint type:",c2026.get("checkpoint_type"))
print("Seed2026 resume epoch/batch:",c2026.get("resume_epoch"),c2026.get("resume_batch"))
print("Expected historical resume point was epoch 35, batch 400.")

# Seed 42 must be generated with train_all_variants.py using mode='signed', seed=42.
print("For Seed42 run: run('signed', 42, output_root='/kaggle/working/SVTR_SIGNED_CHECKPOINT_RECOVERY')")
print("For Seed2026, continue the copied checkpoint with the same optimizer/scheduler/RNG state.")

# Final frozen checkpoint expectations after the recovery actually used for the paper:
FROZEN={
    42: {"best_epoch":40,"val_cer_percent":10.0159},
    123:{"best_epoch":38,"val_cer_percent":9.5512},
    2026:{"best_epoch":38,"val_cer_percent":9.5127},
}
print("Frozen final set:",FROZEN)
print("TEST EVALUATED IN THIS FILE: NO")
