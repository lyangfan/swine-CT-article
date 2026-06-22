#!/usr/bin/env python3
"""Build cases_csv for the locked evaluator (evaluate_swine_ct.py).

For each test case, generates a row with:
  case_id, fold, source, cohort, image_path, label_path, pred_path,
  head_metric_status, testis_metric_status, method

Conditional status:
  HZAU case → head=present_for_dice, testis=absent_or_out_of_fov_for_fp
  TB case   → head=absent_or_out_of_fov_for_fp, testis=present_for_dice

Usage:
    python -m evaluation.build_cases_csv \
        --network nnunet_v1 --seed 20260520 \
        --output-csv evaluation/cases/nnunet_v1__seed20260520.csv
"""
import argparse
import csv
from pathlib import Path

ARTICLE = Path("/home/share/hzau/home/liuyangfan/swine-CT-article")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--case-metadata", default=str(ARTICLE / "data" / "manifests" / "case_metadata.csv"))
    ap.add_argument("--split-manifest", default=str(ARTICLE / "data" / "splits" / "split_manifest.csv"))
    args = ap.parse_args()

    # read test case_ids + sources from split_manifest
    test_cases = []
    with open(args.split_manifest, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                test_cases.append((row["case_id"], row["source"]))

    raw = ARTICLE / "data" / "nnunetv1" / "nnUNet_raw_data" / "Task601_Article622_Carcass9Class"
    pred_dir = ARTICLE / "data" / "nnunetv1" / "v1_comparison_predictions" / f"{args.network}__seed{args.seed}"

    fields = ["case_id", "fold", "source", "cohort", "image_path", "label_path",
              "pred_path", "head_metric_status", "testis_metric_status", "method"]
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for case_id, source in sorted(test_cases):
            head_status = "present_for_dice" if source == "HZAU" else "absent_or_out_of_fov_for_fp"
            testis_status = "absent_or_out_of_fov_for_fp" if source == "HZAU" else "present_for_dice"
            w.writerow({
                "case_id": case_id,
                "fold": 0,
                "source": source,
                "cohort": "v1_comparison",
                "image_path": str(raw / "imagesTs" / f"{case_id}_0000.nii.gz"),
                "label_path": str(raw / "labelsTs" / f"{case_id}.nii.gz"),
                "pred_path": str(pred_dir / f"{case_id}.nii.gz"),
                "head_metric_status": head_status,
                "testis_metric_status": testis_status,
                "method": f"{args.network}_seed{args.seed}",
            })
            n += 1
    print(f"[build_cases_csv] {args.network} seed={args.seed}: {n} test cases -> {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
