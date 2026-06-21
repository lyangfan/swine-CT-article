#!/usr/bin/env python3
"""Build nnU-Net v1 Task601 raw data infrastructure for the article 6:2:2 split.

Layout (Option A — fixed single split):
  imagesTr/labelsTr  = train+val (158)  -> symlinks into labeled_197
  imagesTs/labelsTs  = test (39)        -> symlinks into labeled_197 (frozen, never in training)
  dataset.json       = nnU-Net v1 style, numTraining=158
  splits_final.{json,pkl} = single fold {train: 120, val: 38} from frozen split_manifest.csv
  audit_manifest.json = counts, SHA256, symlink validation, split integrity, source/breed balance

Run on Huawei (paca_share) where labeled_197 resolves. Idempotent (re-run safe).

NEXT STEP (not done here): nnUNet_plan_and_preprocess -t 601, then copy
splits_final.pkl into nnUNet_preprocessed/Task601_.../ (overwrites auto 5-fold).
"""
import csv
import hashlib
import json
import os
import pickle
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ARTICLE_DATA = Path("/home/share/hzau/home/liuyangfan/swine-CT-article/data")
SPLIT_MANIFEST = ARTICLE_DATA / "splits" / "split_manifest.csv"
CASE_META = ARTICLE_DATA / "manifests" / "case_metadata.csv"
LABELED_197 = Path(
    "/home/hzau/whcs-share37/liuyangfan/nnunet_medsam_semisup/data/labeled_197"
)
TASK_DIR = ARTICLE_DATA / "nnunetv1" / "Task601_Article622_Carcass9Class"

LABELS = {
    "0": "background", "1": "front", "2": "middle", "3": "end",
    "4": "left_kidney", "5": "right_kidney", "6": "testis",
    "7": "thoracic_cavity", "8": "abdominal_and_pelvic_cavity", "9": "head",
}


def read_rows(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_symlinks(cases, dst_dir, src_subdir, suffix):
    """suffix='_0000' for images, '' for labels. Link name = {case}{suffix}.nii.gz."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    created, broken = 0, []
    for case in cases:
        src = LABELED_197 / src_subdir / f"{case}.nii.gz"
        link = dst_dir / f"{case}{suffix}.nii.gz"
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(src, link)
        created += 1
        if not src.exists():
            broken.append(str(src))
    return created, broken


def main() -> None:
    split_rows = read_rows(SPLIT_MANIFEST)
    by_split = defaultdict(list)
    for r in split_rows:
        by_split[r["split"]].append(r["case_id"])
    for k in by_split:
        by_split[k].sort()
    train, val, test = by_split["train"], by_split["val"], by_split["test"]
    trainval = sorted(set(train) | set(val))
    assert len(trainval) == len(train) + len(val), "train/val overlap!"

    # case metadata for balance audit
    meta = {r["case_id"]: r for r in read_rows(CASE_META)}

    # --- symlinks ---
    n_img_tr, brk_img_tr = make_symlinks(trainval, TASK_DIR / "imagesTr", "images", "_0000")
    n_lbl_tr, brk_lbl_tr = make_symlinks(trainval, TASK_DIR / "labelsTr", "labels", "")
    n_img_ts, brk_img_ts = make_symlinks(test, TASK_DIR / "imagesTs", "images", "_0000")
    n_lbl_ts, brk_lbl_ts = make_symlinks(test, TASK_DIR / "labelsTs", "labels", "")
    all_broken = brk_img_tr + brk_lbl_tr + brk_img_ts + brk_lbl_ts

    # --- dataset.json (v1 style) ---
    dataset = {
        "name": "Task601_Article622_Carcass9Class",
        "description": "Article fixed 6:2:2 split; train+val=158 in Tr, test=39 held-out in Ts",
        "tensorImageSize": "3D",
        "reference": "swine-CT-article",
        "licence": "internal",
        "release": "2026-06-21",
        "modality": {"0": "CT"},
        "labels": LABELS,
        "numTraining": len(trainval),
        "numTest": len(test),
        "training": [
            {"image": f"./imagesTr/{c}.nii.gz", "label": f"./labelsTr/{c}.nii.gz"}
            for c in trainval
        ],
        "test": list(test),
    }
    dsj_path = TASK_DIR / "dataset.json"
    dsj_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")

    # --- splits_final: single fold (Option A) ---
    splits = [{"train": train, "val": val}]
    (TASK_DIR / "splits_final.json").write_text(
        json.dumps(splits, indent=2), encoding="utf-8"
    )
    with (TASK_DIR / "splits_final.pkl").open("wb") as f:
        pickle.dump(splits, f)

    # --- audit manifest ---
    def bal(case_ids, key):
        c = Counter(meta[c][key] for c in case_ids if c in meta)
        return dict(sorted(c.items()))

    audit = {
        "task": "Task601_Article622_Carcass9Class",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split_source": str(SPLIT_MANIFEST),
        "labeled_197_root": str(LABELED_197),
        "counts": {
            "numTraining": len(trainval),
            "numTest": len(test),
            "train": len(train), "val": len(val), "test": len(test),
            "imagesTr": n_img_tr, "labelsTr": n_lbl_tr,
            "imagesTs": n_img_ts, "labelsTs": n_lbl_ts,
        },
        "dataset_json_sha256": sha256_file(dsj_path),
        "symlink_validation": {
            "broken_targets": all_broken,
            "n_broken": len(all_broken),
        },
        "split_integrity": {
            "train_val_disjoint": len(set(train) & set(val)) == 0,
            "train_test_disjoint": len(set(train) & set(test)) == 0,
            "val_test_disjoint": len(set(val) & set(test)) == 0,
            "union_is_197": len(set(train) | set(val) | set(test)) == 197,
        },
        "balance": {
            "train": {"source": bal(train, "source"), "breed_en": bal(train, "breed_en")},
            "val": {"source": bal(val, "source"), "breed_en": bal(val, "breed_en")},
            "test": {"source": bal(test, "source"), "breed_en": bal(test, "breed_en")},
        },
        "class_presence": {
            "note": "head evaluable on HZAU, testis evaluable on TB",
            "train": {"head_hzau": bal(train, "source").get("HZAU", 0),
                      "testis_tb": bal(train, "source").get("TB", 0)},
            "val": {"head_hzau": bal(val, "source").get("HZAU", 0),
                    "testis_tb": bal(val, "source").get("TB", 0)},
            "test": {"head_hzau": bal(test, "source").get("HZAU", 0),
                     "testis_tb": bal(test, "source").get("TB", 0)},
        },
        "next_step": (
            "Run nnUNet_plan_and_preprocess -t 601, then copy splits_final.pkl "
            "into nnUNet_preprocessed/Task601_Article622_Carcass9Class/ "
            "(overwrites auto 5-fold). Reuse shared_unpacked_cache for .npy."
        ),
    }
    (TASK_DIR / "audit_manifest.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    print(f"task dir: {TASK_DIR}")
    print(f"imagesTr={n_img_tr} labelsTr={n_lbl_tr} | imagesTs={n_img_ts} labelsTs={n_lbl_ts}")
    print(f"broken symlinks: {len(all_broken)}")
    print(f"dataset.json sha256: {audit['dataset_json_sha256']}")
    print(f"split integrity: {audit['split_integrity']}")
    if all_broken:
        print("BROKEN TARGETS:")
        for b in all_broken:
            print(f"  {b}")


if __name__ == "__main__":
    main()
