#!/usr/bin/env python3
"""One-shot test evaluation for a frozen best ROI presence classifier.

Implements spec Â§9: load the frozen best checkpoint (Â§8), predict on test
projections, compute metrics (AUPRC/AUROC/Brier/ECE/FA/FP/sensitivity/
specificity) at threshold 0.5, calibrate temperature, and write
test_metrics.json + test_predictions.csv.

Imports _build_resnet18, _metrics, _temperature_summary, RoiProjectionDataset
from train_roi_classifier.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

# ---- Import from sibling train_roi_classifier (same package) ----------
try:
    from .train_roi_classifier import (
        _build_resnet18,
        _metrics,
        _temperature_summary,
        RoiProjectionDataset,
    )
    from .common import bool_from_row, read_csv
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_roi_classifier import (  # type: ignore
        _build_resnet18,
        _metrics,
        _temperature_summary,
        RoiProjectionDataset,
    )
    from common import bool_from_row, read_csv  # type: ignore


def _load_test_rows(
    projection_manifest: str,
    manifest: str,
    endpoint: str,
    roi_role: str,
    variant: str,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Load test-only projection rows and labels, matching the checkpoint's
    endpoint/roi_role/variant."""
    proj_rows = read_csv(projection_manifest)
    test_rows = [
        r
        for r in proj_rows
        if r.get("endpoint") == endpoint
        and r.get("roi_role") == roi_role
        and r.get("variant") == variant
        and r.get("artifact_role") == "training_variant"
        and r.get("split") == "test"
    ]
    if not test_rows:
        raise ValueError(
            f"No test projection rows found for {endpoint}/{roi_role}/{variant}"
        )

    label_rows = read_csv(manifest)
    label_by_case: dict[str, int] = {}
    for r in label_rows:
        if r.get("split") != "test" or endpoint not in r:
            continue
        label_by_case[r["case_id"]] = bool_from_row(r[endpoint])

    for row in test_rows:
        if row["case_id"] not in label_by_case:
            raise ValueError(
                f"Projection row has no test label: {row['case_id']}"
            )

    return test_rows, label_by_case


def _run_inference(model, loader, device) -> tuple[list[int], list[float], list[str]]:
    """Run inference (no gradients, eval mode). Returns (y_true, y_prob, case_ids)."""
    import torch

    model.eval()
    all_y: list[int] = []
    all_p: list[float] = []
    all_ids: list[str] = []
    with torch.no_grad():
        for x, y, ids in loader:
            x = x.to(device)
            logits = model(x)
            all_y += [int(v) for v in y.cpu().reshape(-1).tolist()]
            all_p += [
                float(v)
                for v in torch.sigmoid(logits).cpu().reshape(-1).tolist()
            ]
            all_ids += list(ids)
    return all_y, all_p, all_ids


def _write_test_predictions_csv(
    path: Path,
    case_ids: list[str],
    labels: list[int],
    probs: list[float],
    endpoint: str,
) -> None:
    """Write test_predictions.csv with columns case_id, <endpoint>, <endpoint>_prob_raw."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case_id", endpoint, f"{endpoint}_prob_raw"],
        )
        writer.writeheader()
        for cid, y, p in zip(case_ids, labels, probs):
            writer.writerow(
                {
                    "case_id": cid,
                    endpoint: int(y),
                    f"{endpoint}_prob_raw": float(p),
                }
            )


def _worker_init_fn(worker_id: int) -> None:
    import numpy as np
    import random
    import torch

    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, Dataset

    # ---- 1. Load checkpoint -----------------------------------------------
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    endpoint = ckpt["endpoint"]
    roi_role = ckpt["roi_role"]
    variant = ckpt["variant"]
    init_mode = ckpt["init_mode"]

    # ---- 2. Load test rows -----------------------------------------------
    test_rows, label_by_case = _load_test_rows(
        args.projection_manifest,
        args.manifest,
        endpoint,
        roi_role,
        variant,
    )

    # ---- 3. Build model and load state dict ------------------------------
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = _build_resnet18(init_mode).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # ---- 4. Create test DataLoader ---------------------------------------
    # RoiProjectionDataset doesn't inherit from torch Dataset; follow the
    # same pattern as train_roi_classifier.
    ds_cls = type(
        "RoiProjectionTestDataset", (RoiProjectionDataset, Dataset), {}
    )
    test_ds = ds_cls(
        test_rows,
        label_by_case,
        init_mode,
        normalize_stats=None,  # ImageNet normalization is applied internally when init_mode=="imagenet"
        augment=False,
    )
    generator = torch.Generator().manual_seed(20260520)
    test_loader = DataLoader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        generator=generator,
        worker_init_fn=_worker_init_fn,
    )

    # ---- 5. Run inference ------------------------------------------------
    y_true, y_prob, case_ids = _run_inference(model, test_loader, device)

    # ---- 6. Compute metrics ----------------------------------------------
    metrics = _metrics(y_true, y_prob)
    calibration = _temperature_summary(y_true, y_prob)

    # ---- 7. Write outputs ------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # test_metrics.json
    test_metrics = {
        "checkpoint": str(checkpoint_path.resolve()),
        "endpoint": endpoint,
        "roi_role": roi_role,
        "variant": variant,
        "init_mode": init_mode,
        "checkpoint_role": ckpt.get("checkpoint_role", "unknown"),
        "epoch": ckpt.get("epoch", -1),
        "test_cases": len(test_rows),
        "metrics": {
            "auprc": metrics.get("auprc"),
            "auroc": metrics.get("auroc"),
            "brier": metrics.get("brier"),
            "ece_10bin": metrics.get("ece_10bin"),
            "false_absent_count": metrics.get("false_absent_count"),
            "false_present_count": metrics.get("false_present_count"),
            "sensitivity": metrics.get("sensitivity"),
            "specificity": metrics.get("specificity"),
            "present_count": metrics.get("present_count"),
            "absent_count": metrics.get("absent_count"),
            "tp": metrics.get("tp"),
            "tn": metrics.get("tn"),
            "false_absent_rate": metrics.get("false_absent_rate"),
            "false_present_rate": metrics.get("false_present_rate"),
        },
        "temperature_calibration": calibration,
    }
    metrics_path = output_dir / "test_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    # test_predictions.csv
    preds_path = output_dir / "test_predictions.csv"
    _write_test_predictions_csv(preds_path, case_ids, y_true, y_prob, endpoint)

    return {
        "status": "PASS",
        "endpoint": endpoint,
        "checkpoint": str(checkpoint_path.resolve()),
        "test_metrics_path": str(metrics_path.resolve()),
        "test_predictions_path": str(preds_path.resolve()),
        "test_cases": len(test_rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to frozen best .pt checkpoint",
    )
    parser.add_argument(
        "--projection-manifest",
        required=True,
        help="Path to ROI projection manifest CSV (must contain test rows)",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to classifier split manifest CSV",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for test_metrics.json and test_predictions.csv",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference (default: 16)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (default: 4)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device (default: cuda if available, else cpu)",
    )
    args = parser.parse_args(argv)

    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
