#!/usr/bin/env python3
"""Shared constants, helpers, and path resolution for roi_classifier tools.

All tools in this package import from here via `from .common import ...`
with a fallback to `from common import ...` when invoked as a standalone
script (not as part of the package).
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import importlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

# ---------------------------------------------------------------------------
# Path resolution — auto-detect the article repo root from __file__
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
SPEC_PATH = _THIS_FILE.parent.parent.parent  # tools/roi_classifier -> tools -> repo root
DEFAULT_CONFIG_PATH = _THIS_FILE.parent / "default_config.yaml"
DEFAULT_OUTPUT_ROOT = SPEC_PATH / "runs" / "roi_presence_classifier"

# ---------------------------------------------------------------------------
# Canonical constants (aligned with article split + formal spec)
# ---------------------------------------------------------------------------

EXPECTED_CASE_COUNT = 197
ENDPOINTS = ("head_present", "testis_present")
FIRST_ROUND_SPLITS = {"train", "val"}  # article split uses "val" NOT "validation"
RESERVED_TEST_SPLIT = "test"
VARIANTS = (
    "muscle_only_mean",
    "muscle_only_p90",
    "foreground_thickness",
    "bone_only_mip",
    "multi_channel_compact",
)
PROJECTION_VARIANTS = VARIANTS
INIT_MODES = ("imagenet", "random")

ROI_ROLES_BY_ENDPOINT: dict[str, list[str]] = {
    "head_present": ["head_cranial_half"],
    "testis_present": ["testis_caudal_half", "testis_caudal_lower_half"],
}

CORRECT_ROI_ROLES = {
    "head_cranial_half",
    "testis_caudal_half",
    "testis_caudal_lower_half",
}

# Wrong / control ROI roles are empty — sentinel only (D16/D22 decision).
WRONG_OR_CONTROL_ROI_ROLES: dict[str, list[str]] = {}

HEAD_LABEL_ID = 9
TESTIS_LABEL_ID = 6

EXPECTED_MODEL_COUNT = (
    sum(len(v) for v in ROI_ROLES_BY_ENDPOINT.values())
    * len(VARIANTS)
    * len(INIT_MODES)
)  # 3 correct roles × 5 variants × 2 init = 30


# ============================================================================
# Utility functions
# ============================================================================


def utc_stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(config_path)
    return cfg


def resolve_output_root(
    cfg: dict[str, Any] | None = None, override: str | None = None
) -> Path:
    if override:
        return Path(override)
    cfg = cfg or {}
    return Path(cfg.get("project", {}).get("output_root", DEFAULT_OUTPUT_ROOT))


def stage_report_dir(output_root: Path, stamp: str, batch: bool = False) -> Path:
    name = f"batch_{stamp}" if batch and not stamp.startswith("batch_") else stamp
    return output_root / "reports" / name


def manifest_dir(output_root: Path) -> Path:
    return output_root / "data" / "manifests"


def projection_dir(output_root: Path, stamp: str) -> Path:
    return output_root / "data" / "projections" / stamp


def preview_dir(output_root: Path, stamp: str) -> Path:
    return output_root / "data" / "previews" / stamp


def models_dir(output_root: Path, stamp: str) -> Path:
    return output_root / "models" / f"batch_{stamp}"


def jobs_dir(output_root: Path, stamp: str) -> Path:
    return output_root / "jobs" / f"batch_{stamp}"


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    ensure_dirs(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def write_csv(
    path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]
) -> int:
    ensure_dirs(path.parent)
    rows = list(rows)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
    return len(rows)


def read_csv(path: str | os.PathLike[str]) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bool_from_row(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if value is None:
        raise ValueError("missing boolean value")
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "present"}:
        return 1
    if text in {"0", "false", "no", "n", "absent"}:
        return 0
    raise ValueError(f"cannot parse boolean value: {value!r}")


def require_columns(
    rows: list[dict[str, Any]], columns: Iterable[str], source: str
) -> None:
    if not rows:
        raise ValueError(f"{source} is empty")
    available = set(rows[0])
    missing = [col for col in columns if col not in available]
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")


def forbid_test_rows(rows: list[dict[str, Any]], source: str) -> None:
    bad = [
        r.get("case_id", "<unknown>")
        for r in rows
        if str(r.get("split", "")).strip() == RESERVED_TEST_SPLIT
    ]
    if bad:
        raise ValueError(
            f"{source} contains reserved test rows ({len(bad)} cases); "
            "first-round code must not consume test data"
        )


def parse_case_id(path_or_id: str) -> str:
    text = Path(str(path_or_id)).name
    for suffix in (
        ".nii.gz",
        ".nii",
        ".mha",
        ".mhd",
        ".nrrd",
        ".npz",
        ".npy",
        ".pkl",
    ):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return re.sub(r"_0000$", "", text)


def optional_import(name: str, purpose: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(
            f"Missing optional dependency {name!r} required for {purpose}. "
            "Install it in the runtime or use an input format handled without it."
        ) from exc


def set_reproducible_seed(
    seed: int, deterministic_cudnn: bool | None = None
) -> dict[str, Any]:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    info: dict[str, Any] = {"seed": seed}
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn is not None:
            torch.backends.cudnn.deterministic = bool(deterministic_cudnn)
            torch.backends.cudnn.benchmark = not bool(deterministic_cudnn)
        info.update(
            {
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "torch_num_threads": int(torch.get_num_threads()),
                "torch_num_interop_threads": int(torch.get_num_interop_threads()),
            }
        )
    except Exception as exc:
        info["torch_seed_warning"] = repr(exc)
    return info


def model_grid(
    endpoints: Iterable[str] | None = None,
    roi_roles: Iterable[str] | None = None,
    variants: Iterable[str] | None = None,
    init_modes: Iterable[str] | None = None,
) -> list[dict[str, str]]:
    endpoints = tuple(endpoints) if endpoints else ENDPOINTS
    variants = tuple(variants) if variants else VARIANTS
    init_modes = tuple(init_modes) if init_modes else INIT_MODES
    roi_role_filter = set(roi_roles) if roi_roles else None
    rows: list[dict[str, str]] = []
    for endpoint in endpoints:
        if endpoint not in ROI_ROLES_BY_ENDPOINT:
            raise ValueError(f"unknown endpoint: {endpoint}")
        for roi_role in ROI_ROLES_BY_ENDPOINT[endpoint]:
            if roi_role_filter and roi_role not in roi_role_filter:
                continue
            for variant in variants:
                if variant not in VARIANTS:
                    raise ValueError(f"unknown variant: {variant}")
                for init_mode in init_modes:
                    if init_mode not in INIT_MODES:
                        raise ValueError(f"unknown init mode: {init_mode}")
                    rows.append(
                        {
                            "endpoint": endpoint,
                            "roi_role": roi_role,
                            "variant": variant,
                            "init_mode": init_mode,
                        }
                    )
    return rows


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML config path",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override artifact root",
    )
    parser.add_argument(
        "--stamp",
        default=None,
        help="Run stamp; defaults to UTC timestamp",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan only; do not write outputs or submit jobs",
    )


def command_result(status: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"status": status, "message": message, **extra}


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))


def assert_no_reserved_test(path: str | os.PathLike[str], source: str) -> None:
    rows = read_csv(path)
    forbid_test_rows(rows, source)


def this_tool_dir() -> Path:
    return Path(__file__).resolve().parent


def python_executable() -> str:
    return sys.executable


# ---------------------------------------------------------------------------
# Standalone-script fallback
# ---------------------------------------------------------------------------

def _patch_path_for_standalone():
    """Allow `python tools/roi_classifier/<some_tool>.py` to import `common`
    when not run as `-m tools.roi_classifier.<module>`."""
    pkg_dir = str(_THIS_FILE.parent)
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)
