#!/usr/bin/env python3
"""Stage 1: Place our single-fold custom split into the Task601 preprocessed dir.

nnU-Net v1 `plan_and_preprocess` does NOT write splits_final.pkl into the
preprocessed dir; the trainer creates an auto 5-fold split on first run if the
file is absent. We must copy our frozen 6:2:2 split (train=120 / val=38, the
same fold0 for every network/seed) into the preprocessed dir BEFORE training so
all networks read the identical train/val partition.

Source of truth:
  nnUNet_raw_data/Task601_Article622_Carcass9Class/splits_final.pkl
    (written by build_task601.py from data/splits/split_manifest.csv, seed=42)

Destination (overwrites any auto 5-fold):
  nnUNet_preprocessed/Task601_Article622_Carcass9Class/splits_final.pkl

Idempotent: re-running copies again and re-verifies. Run on Huawei (paca_share)
where both paths resolve.

Usage:
  python place_split.py            # default paths (Huawei swine-CT-article root)
  python place_split.py --verify   # only verify, do not copy
"""
import argparse
import pickle
import shutil
from pathlib import Path

DEFAULT_ROOT = Path("/home/share/hzau/home/liuyangfan/swine-CT-article")
TASK = "Task601_Article622_Carcass9Class"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root", default=str(DEFAULT_ROOT),
        help="swine-CT-article root (Huawei).",
    )
    ap.add_argument("--verify", action="store_true", help="only verify, skip copy")
    args = ap.parse_args()

    root = Path(args.root)
    raw_dir = root / "data" / "nnunetv1" / "nnUNet_raw_data" / TASK
    pre_dir = root / "data" / "nnunetv1" / "nnUNet_preprocessed" / TASK
    src = raw_dir / "splits_final.pkl"
    dst = pre_dir / "splits_final.pkl"

    assert src.exists(), f"source split missing: {src}"
    assert pre_dir.exists(), (
        f"preprocessed dir missing: {pre_dir}\n"
        "Run nnUNet_plan_and_preprocess -t 601 first (Stage 0)."
    )

    # --- load + validate the source split ---
    splits = pickle.load(src.open("rb"))
    assert isinstance(splits, list) and len(splits) == 1, (
        f"expected single-fold split, got {len(splits)} folds"
    )
    fold0 = splits[0]
    n_train, n_val = len(fold0["train"]), len(fold0["val"])
    assert n_train == 120 and n_val == 38, (
        f"fold0 must be train=120/val=38, got train={n_train}/val={n_val}"
    )
    train_set, val_set = set(fold0["train"]), set(fold0["val"])
    assert not (train_set & val_set), "train/val overlap!"
    print(f"[ok] source split: 1 fold, train={n_train} val={n_val}, disjoint")

    # --- copy (idempotent) ---
    if args.verify:
        if dst.exists():
            existing = pickle.load(dst.open("rb"))
            same = (
                len(existing) == 1
                and set(existing[0]["train"]) == train_set
                and set(existing[0]["val"]) == val_set
            )
            print(f"[verify] dst exists: {dst} -> {'MATCH' if same else 'MISMATCH'}")
            return 0 if same else 1
        print(f"[verify] dst missing: {dst}")
        return 1

    shutil.copy2(src, dst)
    print(f"[copy] {src}\n   -> {dst}")

    # --- re-verify destination round-trips identically ---
    dst_splits = pickle.load(dst.open("rb"))
    assert len(dst_splits) == 1
    assert set(dst_splits[0]["train"]) == train_set
    assert set(dst_splits[0]["val"]) == val_set
    print("[ok] destination verified: train/val sets identical to source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
