#!/usr/bin/env python3
"""Train endpoint-specific binary ResNet-18 ROI presence classifiers.

Ported from AutoScientists swct06042040 to swine-CT-article (§7 mandated changes):
  - Grid = 30 models (3 correct roles x 5 variants x 2 init). Only correct ROI roles.
  - Use "val" NOT "validation" throughout (split filtering, path naming).
  - Reads classifier_split_manifest.csv.
  - Training seed: 20260520 (single seed).
  - Per-model best = val AUPRC highest subject to FA<=1 safety gate.
  - Hyperparams verbatim D18.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import socket
import sys
from pathlib import Path
from typing import Any

try:
    from .common import (
        EXPECTED_MODEL_COUNT,
        INIT_MODES,
        ROI_ROLES_BY_ENDPOINT,
        VARIANTS,
        add_common_args,
        bool_from_row,
        command_result,
        forbid_test_rows,
        load_config,
        model_grid,
        models_dir,
        print_json,
        read_csv,
        resolve_output_root,
        set_reproducible_seed,
        stage_report_dir,
        utc_stamp,
        write_json,
    )
except ImportError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parent))
    from common import (  # type: ignore
        EXPECTED_MODEL_COUNT,
        INIT_MODES,
        ROI_ROLES_BY_ENDPOINT,
        VARIANTS,
        add_common_args,
        bool_from_row,
        command_result,
        forbid_test_rows,
        load_config,
        model_grid,
        models_dir,
        print_json,
        read_csv,
        resolve_output_root,
        set_reproducible_seed,
        stage_report_dir,
        utc_stamp,
        write_json,
    )


def _lazy_training_imports():
    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from torch.utils.data import DataLoader, Dataset

    return np, torch, DataLoader, Dataset, average_precision_score, brier_score_loss, roc_auc_score


def _build_resnet18(init_mode: str):
    import torch.nn as nn

    try:
        import torchvision.models as models
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torchvision is required for ROI classifier training") from exc
    if init_mode == "imagenet":
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights)
        except Exception:
            model = models.resnet18(pretrained=True)
    elif init_mode == "random":
        model = models.resnet18(weights=None)
    else:
        raise ValueError(f"unknown init_mode: {init_mode}")
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def _ece(labels, probs, bins: int = 10) -> float:
    np, *_ = _lazy_training_imports()
    y = np.asarray(labels, dtype="float64")
    p = np.asarray(probs, dtype="float64")
    ece = 0.0
    for idx in range(bins):
        lo = idx / bins
        hi = (idx + 1) / bins
        mask = (p >= lo) & (p <= hi) if idx == bins - 1 else (p >= lo) & (p < hi)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(ece)


def _metrics(y_true, y_prob) -> dict[str, Any]:
    np, _, _, _, average_precision_score, brier_score_loss, roc_auc_score = _lazy_training_imports()
    y = np.asarray(y_true, dtype="int64")
    p = np.asarray(y_prob, dtype="float64")
    pred = p >= 0.5
    tp = int(((y == 1) & pred).sum())
    tn = int(((y == 0) & (~pred)).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & (~pred)).sum())
    present = int((y == 1).sum())
    absent = int((y == 0).sum())
    out = {
        "present_count": present,
        "absent_count": absent,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": float(tp / present) if present else float("nan"),
        "specificity": float(tn / absent) if absent else float("nan"),
        "false_absent_count": fn,
        "false_present_count": fp,
        "false_absent_rate": float(fn / present) if present else float("nan"),
        "false_present_rate": float(fp / absent) if absent else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "ece_10bin": _ece(y, p, bins=10),
    }
    if len(set(y.tolist())) > 1:
        out["auprc"] = float(average_precision_score(y, p))
        out["auroc"] = float(roc_auc_score(y, p))
    else:
        out["auprc"] = float("nan")
        out["auroc"] = float("nan")
    return out


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _bce_1d(labels, probs) -> float:
    total = 0.0
    for y, p in zip(labels, probs):
        p = min(max(float(p), 1e-6), 1.0 - 1e-6)
        total += -(int(y) * math.log(p) + (1 - int(y)) * math.log(1.0 - p))
    return float(total / max(len(labels), 1))


def _temperature_summary(labels, probs) -> dict[str, Any]:
    logits = [_logit(float(p)) for p in probs]
    best = {"temperature": 1.0, "bce": _bce_1d(labels, probs)}
    for idx in range(10, 101):
        temp = idx / 10.0
        calibrated = [_sigmoid(x / temp) for x in logits]
        bce = _bce_1d(labels, calibrated)
        if bce < float(best["bce"]):
            best = {"temperature": temp, "bce": bce}
    calibrated = [_sigmoid(x / float(best["temperature"])) for x in logits]
    return {
        **best,
        "raw_bce": _bce_1d(labels, probs),
        "raw_ece_10bin": _ece(labels, probs, bins=10),
        "calibrated_ece_10bin": _ece(labels, calibrated, bins=10),
    }


class RoiProjectionDataset:
    def __init__(self, rows, label_by_case, init_mode: str, normalize_stats=None, augment: bool = False):
        self.rows = rows
        self.label_by_case = label_by_case
        self.init_mode = init_mode
        self.normalize_stats = normalize_stats
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        np, torch, *_ = _lazy_training_imports()
        row = self.rows[index]
        arr = np.load(row["array_path"]).astype("float32")
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] != 3:
            raise ValueError(f"expected 3-channel HWC array, got {arr.shape} for {row['array_path']}")
        arr = np.clip(arr, 0.0, 1.0)
        if self.augment:
            if np.random.rand() < 0.5:
                arr = np.clip(arr * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.03, 0.03), 0, 1)
        if self.init_mode == "imagenet":
            mean = np.asarray([0.485, 0.456, 0.406], dtype="float32")
            std = np.asarray([0.229, 0.224, 0.225], dtype="float32")
            arr = (arr - mean) / std
        else:
            if self.normalize_stats is None:
                raise ValueError("random-init mode requires train-split normalization stats")
            mean = self.normalize_stats["mean"]
            std = self.normalize_stats["std"]
            arr = (arr - mean) / np.maximum(std, 1e-6)
        arr = np.ascontiguousarray(arr.transpose(2, 0, 1))
        y = self.label_by_case[row["case_id"]]
        return torch.from_numpy(arr), torch.tensor([y], dtype=torch.float32), row["case_id"]


def _train_stats(rows):
    np, *_ = _lazy_training_imports()
    sums = np.zeros(3, dtype="float64")
    sqs = np.zeros(3, dtype="float64")
    count = 0
    for row in rows:
        arr = np.load(row["array_path"]).astype("float32")
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = np.clip(arr, 0, 1)
        flat = arr.reshape(-1, 3)
        sums += flat.sum(axis=0)
        sqs += (flat**2).sum(axis=0)
        count += flat.shape[0]
    mean = sums / max(count, 1)
    var = sqs / max(count, 1) - mean**2
    return {"mean": mean.astype("float32"), "std": np.sqrt(np.maximum(var, 1e-8)).astype("float32")}


def _worker_init_fn(worker_id: int) -> None:
    import numpy as np
    import random
    import torch

    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def _load_rows(projection_manifest: str, manifest: str, endpoint: str, roi_role: str, variant: str):
    rows = read_csv(projection_manifest)
    if any(r.get("split") == "test" for r in rows):
        raise ValueError("ROI projection manifest contains reserved test rows; refusing first-round training")
    rows = [
        r
        for r in rows
        if r.get("endpoint") == endpoint
        and r.get("roi_role") == roi_role
        and r.get("variant") == variant
        and r.get("artifact_role") == "training_variant"
        and r.get("split") in {"train", "val"}
    ]
    forbid_test_rows(rows, "ROI projection manifest")
    labels = read_csv(manifest)
    label_by_case = {
        r["case_id"]: bool_from_row(r[endpoint])
        for r in labels
        if r.get("split") in {"train", "val"} and endpoint in r
    }
    for row in rows:
        if row["case_id"] not in label_by_case:
            raise ValueError(f"projection row has no train/val label: {row['case_id']}")
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    if not train_rows or not val_rows:
        raise ValueError(f"missing train/val rows for {endpoint}/{roi_role}/{variant}")
    return train_rows, val_rows, label_by_case


def _pos_weight(train_rows, label_by_case, cap: float):
    np, torch, *_ = _lazy_training_imports()
    y = np.asarray([label_by_case[r["case_id"]] for r in train_rows], dtype="float32")
    pos = float(y.sum())
    neg = float(y.shape[0] - pos)
    if pos == 0 or neg == 0:
        raise ValueError(f"invalid train split for pos_weight: pos={pos}, neg={neg}")
    raw = neg / pos
    capped = min(raw, cap)
    return torch.tensor([capped], dtype=torch.float32), raw, capped


def _run_epoch(model, loader, loss_fn, optimizer, scaler, device, train: bool, amp: bool):
    import torch

    model.train(train)
    losses = []
    all_y = []
    all_p = []
    case_ids = []
    for x, y, ids in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits = model(x)
                loss = loss_fn(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        losses.append(float(loss.detach().cpu()))
        all_y += [int(v) for v in y.detach().cpu().reshape(-1).tolist()]
        all_p += [float(v) for v in torch.sigmoid(logits.detach()).cpu().reshape(-1).tolist()]
        case_ids += list(ids)
    return float(sum(losses) / max(len(losses), 1)), all_y, all_p, case_ids


def _write_prediction_csv(path: Path, case_ids: list[str], labels: list[int], probs: list[float], endpoint: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", endpoint, f"{endpoint}_prob_raw"])
        writer.writeheader()
        for cid, y, p in zip(case_ids, labels, probs):
            writer.writerow({"case_id": cid, endpoint: int(y), f"{endpoint}_prob_raw": float(p)})


def _runtime_context(args: argparse.Namespace, cfg: dict[str, Any], stamp: str) -> dict[str, Any]:
    context: dict[str, Any] = {
        "stamp": stamp,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "resource_request": args.resource_request,
        "node_constraint": args.node_constraint,
        "job_name": args.job_name,
        "job_id_arg": args.job_id,
        "env": {
            "DSUB_JOB_ID": os.environ.get("DSUB_JOB_ID"),
            "JOB_ID": os.environ.get("JOB_ID"),
            "TASK_EXEC_NODES": os.environ.get("TASK_EXEC_NODES"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "training_policy": {
            "architecture": cfg["model_grid"]["architecture"],
            "classifier_type": cfg["model_grid"]["classifier_type"],
            "seed": cfg["training"]["seed"],
            "max_epochs": args.max_epochs or cfg["training"]["max_epochs"],
            "loss": cfg["training"]["loss"],
            "pos_weight_cap": cfg["training"]["pos_weight_cap"],
            "amp": bool(args.amp),
            "deterministic_cudnn": bool(cfg["training"].get("deterministic_cudnn", True)),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
        },
    }
    try:
        import torch

        context["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "num_threads": int(torch.get_num_threads()),
            "num_interop_threads": int(torch.get_num_interop_threads()),
        }
    except Exception as exc:
        context["torch_error"] = repr(exc)
    return context


def train_one(args, cfg: dict[str, Any], grid_item: dict[str, str], stamp: str, output_root: Path) -> dict[str, Any]:
    np, torch, DataLoader, Dataset, *_ = _lazy_training_imports()
    endpoint = grid_item["endpoint"]
    roi_role = grid_item["roi_role"]
    variant = grid_item["variant"]
    init_mode = grid_item["init_mode"]
    seed_info = set_reproducible_seed(int(cfg["training"]["seed"]), bool(cfg["training"].get("deterministic_cudnn", True)))
    train_rows, val_rows, label_by_case = _load_rows(args.projection_manifest, args.manifest, endpoint, roi_role, variant)
    normalize_stats = _train_stats(train_rows) if init_mode == "random" else None
    ds_base = type("RoiProjectionTorchDataset", (RoiProjectionDataset, Dataset), {})
    train_ds = ds_base(train_rows, label_by_case, init_mode, normalize_stats, augment=not args.no_augment)
    train_eval_ds = ds_base(train_rows, label_by_case, init_mode, normalize_stats, augment=False)
    val_ds = ds_base(val_rows, label_by_case, init_mode, normalize_stats, augment=False)
    generator = torch.Generator().manual_seed(int(cfg["training"]["seed"]))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        generator=generator,
        worker_init_fn=_worker_init_fn,
    )
    train_eval_loader = DataLoader(train_eval_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
    pos_weight, raw_pw, capped_pw = _pos_weight(train_rows, label_by_case, float(cfg["training"]["pos_weight_cap"]))
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _build_resnet18(init_mode).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    lr = float(cfg["training"]["lr"][init_mode])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(cfg["training"]["weight_decay"]))
    max_epochs = int(args.max_epochs or cfg["training"]["max_epochs"])
    warmup = int(cfg["training"]["warmup_epochs"])

    def lr_lambda(epoch):
        if epoch < warmup:
            return float(epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(max_epochs - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    out_dir = models_dir(output_root, stamp) / f"{endpoint}_{roi_role}_{variant}_{init_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "training_history.csv"
    best = {"epoch": -1, "auprc": -1.0, "safety_pass": False, "state": None, "metrics": None}
    stale = 0
    patience = int(cfg["training"]["early_stop_patience"])
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_auprc",
                "val_auroc",
                "val_false_absent_count",
                "val_false_present_count",
                "val_sensitivity",
                "val_specificity",
                "safety_pass",
            ],
        )
        writer.writeheader()
        for epoch in range(max_epochs):
            train_loss, _, _, _ = _run_epoch(model, train_loader, loss_fn, optimizer, scaler, device, True, args.amp)
            val_loss, y_true, y_prob, _ = _run_epoch(model, val_loader, loss_fn, None, None, device, False, args.amp)
            scheduler.step()
            metrics = _metrics(y_true, y_prob)
            safety = int(metrics["false_absent_count"]) <= int(cfg["training"]["false_absent_gate_default"])
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_auprc": metrics["auprc"],
                    "val_auroc": metrics["auroc"],
                    "val_false_absent_count": metrics["false_absent_count"],
                    "val_false_present_count": metrics["false_present_count"],
                    "val_sensitivity": metrics["sensitivity"],
                    "val_specificity": metrics["specificity"],
                    "safety_pass": int(safety),
                }
            )
            score = float(metrics["auprc"]) if not math.isnan(float(metrics["auprc"])) else -1.0
            better = (safety and not best["safety_pass"]) or (safety == best["safety_pass"] and score > float(best["auprc"]))
            if better:
                best = {
                    "epoch": epoch,
                    "auprc": score,
                    "safety_pass": bool(safety),
                    "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "metrics": metrics,
                }
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
    final_epoch = int(epoch)
    final_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(
        {
            "model_state_dict": final_state,
            **grid_item,
            "epoch": final_epoch,
            "checkpoint_role": "final_epoch",
            "seed_info": seed_info,
        },
        out_dir / "model_final.pt",
    )
    if best["state"] is None:
        best["state"] = final_state
        best["metrics"] = {}
    model.load_state_dict(best["state"])
    _, train_y, train_p, train_ids = _run_epoch(model, train_eval_loader, loss_fn, None, None, device, False, args.amp)
    _, val_y, val_p, val_ids = _run_epoch(model, val_loader, loss_fn, None, None, device, False, args.amp)
    train_metrics = _metrics(train_y, train_p)
    val_metrics = _metrics(val_y, val_p)
    calibration = _temperature_summary(val_y, val_p)
    _write_prediction_csv(out_dir / "train_predictions.csv", train_ids, train_y, train_p, endpoint)
    _write_prediction_csv(out_dir / "val_predictions.csv", val_ids, val_y, val_p, endpoint)
    torch.save(
        {
            "model_state_dict": best["state"],
            **grid_item,
            "epoch": best["epoch"],
            "checkpoint_role": "best_val_auprc_under_false_absent_safety_gate",
            "best": best,
            "seed_info": seed_info,
        },
        out_dir / "model_best.pt",
    )
    summary = {
        **grid_item,
        "train_cases": len(train_rows),
        "val_cases": len(val_rows),
        "final_epoch": final_epoch,
        "best_epoch": best["epoch"],
        "best_auprc": best["auprc"],
        "best_safety_pass": best["safety_pass"],
        "best_metrics": best["metrics"],
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "val_calibration": calibration,
        "pos_weight_raw": raw_pw,
        "pos_weight_capped": capped_pw,
        "normalization_stats": None if normalize_stats is None else {k: v.tolist() for k, v in normalize_stats.items()},
        "seed_info": seed_info,
        "output_dir": str(out_dir),
    }
    write_json(out_dir / "metrics_summary.json", summary)
    write_json(
        out_dir / "run_manifest.json",
        {
            "schema": "roi_half_projection_presence_classifier_model_run_manifest_v1",
            "stamp": stamp,
            **grid_item,
            "architecture": "resnet18",
            "classifier_type": "endpoint_specific_binary",
            "train_cases": len(train_rows),
            "val_cases": len(val_rows),
            "test_consumed": False,
            "checkpoint_policy": {
                "best": "model_best.pt is best val AUPRC subject to false-absent safety gate",
                "final": "model_final.pt is final epoch checkpoint",
                "best_epoch": best["epoch"],
                "final_epoch": final_epoch,
            },
            "files": {
                "metrics_summary": str(out_dir / "metrics_summary.json"),
                "training_history": str(history_path),
                "train_predictions": str(out_dir / "train_predictions.csv"),
                "val_predictions": str(out_dir / "val_predictions.csv"),
                "model_best": str(out_dir / "model_best.pt"),
                "model_final": str(out_dir / "model_final.pt"),
            },
            "policy": {
                "loss": "BCEWithLogitsLoss with capped train-only pos_weight",
                "seed": int(cfg["training"]["seed"]),
                "amp": bool(args.amp),
                "deterministic_cudnn": bool(cfg["training"].get("deterministic_cudnn", True)),
                "sealed_test_policy": "not consumed",
            },
        },
    )
    return summary


def _write_reports(rdir: Path, plan: dict[str, Any], results: list[dict[str, Any]], runtime_context: dict[str, Any]) -> None:
    rdir.mkdir(parents=True, exist_ok=True)
    metrics_lines = [
        "# ROI Classifier Metrics",
        "",
        "Train/val only. Each row is one endpoint-specific binary model.",
        "",
        "| endpoint | roi_role | variant | init | split | AUPRC | AUROC | sensitivity | specificity | FA | FP | Brier | ECE10 |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        for split_name, metrics in (("train", result["train_metrics"]), ("val", result["val_metrics"])):
            metrics_lines.append(
                "| {endpoint} | {roi_role} | {variant} | {init_mode} | {split_name} | {auprc:.4f} | {auroc:.4f} | {sens:.4f} | {spec:.4f} | {fa} | {fp} | {brier:.4f} | {ece:.4f} |".format(
                    endpoint=result["endpoint"],
                    roi_role=result["roi_role"],
                    variant=result["variant"],
                    init_mode=result["init_mode"],
                    split_name=split_name,
                    auprc=float(metrics["auprc"]),
                    auroc=float(metrics["auroc"]),
                    sens=float(metrics["sensitivity"]),
                    spec=float(metrics["specificity"]),
                    fa=metrics["false_absent_count"],
                    fp=metrics["false_present_count"],
                    brier=float(metrics["brier"]),
                    ece=float(metrics["ece_10bin"]),
                )
            )
    (rdir / "classifier_metrics.md").write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")
    write_json(
        rdir / "training_summary.json",
        {
            **plan,
            "status": "PASS",
            "runtime_context": runtime_context,
            "model_count": len(results),
            "expected_model_count": EXPECTED_MODEL_COUNT,
            "model_outputs": results,
            "no_test_metrics": True,
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    output_root = resolve_output_root(cfg, args.output_root)
    stamp = args.stamp or utc_stamp()
    endpoints = args.endpoints.split(",") if args.endpoints else None
    roi_roles = args.roi_roles.split(",") if args.roi_roles else None
    variants = args.variants.split(",") if args.variants else None
    init_modes = args.init_modes.split(",") if args.init_modes else None
    grid = model_grid(endpoints, roi_roles, variants, init_modes)
    plan = command_result(
        "DRY_RUN" if args.dry_run else "WRITE",
        "ROI classifier training plan validated",
        output_root=str(output_root),
        stamp=stamp,
        model_count=len(grid),
        full_grid=len(grid) == EXPECTED_MODEL_COUNT,
        expected_model_count=EXPECTED_MODEL_COUNT,
        projection_manifest=args.projection_manifest,
        manifest=args.manifest,
        grid=grid,
        test_consumed=False,
    )
    if args.dry_run:
        return plan
    results = [train_one(args, cfg, item, stamp, output_root) for item in grid]
    rdir = stage_report_dir(output_root, stamp)
    runtime_context = _runtime_context(args, cfg, stamp)
    _write_reports(rdir, plan, results, runtime_context)
    return {**plan, "status": "PASS", "trained_models": len(results), "summary_path": str(rdir / "training_summary.json")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--repo-root", default=None, help="Article repo root; defaults to auto-detected path")
    parser.add_argument("--manifest", default=None, help="Classifier split manifest CSV; default <repo-root>/data/manifests/classifier_split_manifest.csv")
    parser.add_argument("--projection-manifest", required=True, help="R3 ROI projection manifest")
    parser.add_argument("--endpoints", default=None, help="Comma-separated endpoints; default full first-round grid")
    parser.add_argument("--roi-roles", default=None, help="Comma-separated ROI roles; default all roles for selected endpoints")
    parser.add_argument("--variants", default=None, help="Comma-separated variants; default fixed five variants")
    parser.add_argument("--init-modes", default=None, help="Comma-separated init modes: imagenet,random")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", help="Enable AMP on CUDA")
    parser.add_argument("--no-augment", action="store_true", help="Disable allowed light 2D augmentations")
    parser.add_argument("--resource-request", default="", help="Scheduler resource request recorded in runtime manifest")
    parser.add_argument("--node-constraint", default="", help="Submitted Huawei node constraint recorded in runtime manifest")
    parser.add_argument("--job-name", default="", help="Scheduler job name recorded in runtime manifest")
    parser.add_argument("--job-id", default="", help="Scheduler job id recorded in runtime manifest when known")
    args = parser.parse_args(argv)

    # Resolve --repo-root and --manifest defaults
    from common import SPEC_PATH  # noqa: F811
    repo_root = Path(args.repo_root) if args.repo_root else SPEC_PATH
    if not args.manifest:
        args.manifest = str(repo_root / "data" / "manifests" / "classifier_split_manifest.csv")

    print_json(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
