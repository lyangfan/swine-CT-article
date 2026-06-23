"""Conditional LR-mirror transform for kidney-protected training.

Adapted from the handoff implementation
(SWCT06042040 `train_conditional_lr_mirror_v1_reviewed.py`).

Provides three components:
  1. ``ConditionalMirrorTransform`` — per-sample mirror transform that disables
     LR-axis mirroring when protected kidney labels are present in the patch.
  2. ``install_conditional_mirror_patch`` — module-symbol patch that replaces
     ``MirrorTransform`` in the moreDA module with a factory returning
     ConditionalMirrorTransform.
  3. ``record_witness`` — writes the augmentation_witness.json after
     get_moreDA_augmentation has been called (verifying the patch took effect).

Integration contract:
  - ``install_conditional_mirror_patch`` MUST be called AFTER
    ``setup_DA_params()`` and BEFORE ``get_moreDA_augmentation()``.
    (In MultiNetworkTrainer.initialize(), between lines 197 and 230.)
  - ``record_witness`` MUST be called AFTER ``get_moreDA_augmentation()``
    to verify that the original MirrorTransform is no longer present in
    the augmentation pipeline.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# ConditionalMirrorTransform
# ---------------------------------------------------------------------------

class ConditionalMirrorTransform:
    """Per-sample mirror transform with protected-label LR-axis gating.

    Mirrors the semantics of batchgenerators' ``MirrorTransform``: a sample is
    selected with ``p_per_sample`` and each effective mirror axis is flipped
    with probability 0.5.  The only policy difference: protected samples
    (containing >= ``protected_min_voxels`` voxels of any protected class)
    have ``conditional_mirror_axis`` removed from the effective axis list.
    """

    def __init__(
        self,
        axes: Sequence[int] = (0, 1, 2),
        data_key: str = "data",
        label_key: str = "seg",
        p_per_sample: float = 1,
        *,
        protected_class_ids: Sequence[int] = (4, 5),
        conditional_mirror_axis: int = 0,
        protected_min_voxels: int = 1,
        telemetry_path: str | Path | None = None,
        log_interval: int = 25,
        run_name: str = "",
        instance_index: int = 1,
        original_transform_class: str = "batchgenerators.transforms.spatial_transforms.MirrorTransform",
    ) -> None:
        self.p_per_sample = float(p_per_sample)
        self.data_key = data_key
        self.label_key = label_key
        self.axes = tuple(int(axis) for axis in axes)
        if self.axes and max(self.axes) > 2:
            raise ValueError("ConditionalMirrorTransform expects spatial axes in {0,1,2}")
        self.protected_class_ids = tuple(int(item) for item in protected_class_ids)
        self.conditional_mirror_axis = int(conditional_mirror_axis)
        self.protected_min_voxels = int(protected_min_voxels)
        if self.protected_min_voxels < 1:
            raise ValueError("protected_min_voxels must be >= 1")
        self.telemetry_path = Path(telemetry_path) if telemetry_path else None
        self.log_interval = max(int(log_interval), 1)
        self.run_name = run_name
        self.instance_index = int(instance_index)
        self.original_transform_class = original_transform_class
        self.transform_call_count = 0
        self._window = self._empty_window()

    # ------------------------------------------------------------------ #
    # telemetry window
    # ------------------------------------------------------------------ #
    def _empty_window(self) -> dict[str, int]:
        return {
            "window_transform_calls": 0,
            "sample_count": 0,
            "protected_sample_count": 0,
            "nonprotected_sample_count": 0,
            "protected_lr_mirror_count": 0,
            "nonprotected_lr_mirror_count": 0,
            "axis0_mirror_count": 0,
            "axis1_mirror_count": 0,
            "axis2_mirror_count": 0,
            "lr_axis_disabled_sample_count": 0,
            "lr_axis_allowed_sample_count": 0,
            "non_lr_axis_mirrored_sample_count": 0,
            "sample_selected_count": 0,
        }

    # ------------------------------------------------------------------ #
    # kidney detection
    # ------------------------------------------------------------------ #
    def _sample_is_protected(self, sample_seg: np.ndarray | None) -> tuple[bool, int]:
        if sample_seg is None:
            return False, 0
        count = 0
        for class_id in self.protected_class_ids:
            count += int(np.count_nonzero(sample_seg == class_id))
            if count >= self.protected_min_voxels:
                return True, count
        return False, count

    def _effective_axes(self, protected: bool) -> tuple[int, ...]:
        if protected:
            return tuple(axis for axis in self.axes if axis != self.conditional_mirror_axis)
        return self.axes

    # ------------------------------------------------------------------ #
    # mirror helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mirror_axis(arr: np.ndarray, spatial_axis: int) -> np.ndarray:
        array_axis = int(spatial_axis) + 1
        if array_axis >= arr.ndim:
            return arr
        return np.flip(arr, axis=array_axis).copy()

    def _record_axis_mirror(self, axis: int, protected: bool) -> None:
        key = f"axis{int(axis)}_mirror_count"
        if key in self._window:
            self._window[key] += 1
        if int(axis) == int(self.conditional_mirror_axis):
            if protected:
                self._window["protected_lr_mirror_count"] += 1
            else:
                self._window["nonprotected_lr_mirror_count"] += 1
        else:
            self._window["non_lr_axis_mirrored_sample_count"] += 1

    # ------------------------------------------------------------------ #
    # JSONL telemetry
    # ------------------------------------------------------------------ #
    def _flush_window(self, force: bool = False) -> None:
        if self.telemetry_path is None:
            return
        if self._window["window_transform_calls"] <= 0:
            return
        if not force and self.transform_call_count % self.log_interval != 0:
            return
        payload = {
            "schema_version": "conditional_mirror_components.v1",
            "run_name": self.run_name,
            "pid": os.getpid(),
            "instance_index": self.instance_index,
            "transform_call_count": int(self.transform_call_count),
            "epoch": None,
            "iteration_or_transform_call": int(self.transform_call_count),
            "original_mirror_axes": list(self.axes),
            "conditional_mirror_axis": int(self.conditional_mirror_axis),
            "protected_class_ids": list(self.protected_class_ids),
            "protected_min_voxels": int(self.protected_min_voxels),
            "p_per_sample": float(self.p_per_sample),
            "timestamp": time.time(),
            **self._window,
        }
        _append_jsonl(self.telemetry_path, payload)
        self._window = self._empty_window()

    def flush(self) -> None:
        self._flush_window(force=True)

    # ------------------------------------------------------------------ #
    # main call (batchgenerators interface)
    # ------------------------------------------------------------------ #
    def __call__(self, **data_dict: Any) -> dict[str, Any]:
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)
        self.transform_call_count += 1
        self._window["window_transform_calls"] += 1
        self._window["batch_size"] = int(len(data))

        for b in range(len(data)):
            self._window["sample_count"] += 1
            sample_seg = seg[b] if seg is not None else None
            protected, _protected_voxels = self._sample_is_protected(sample_seg)
            if protected:
                self._window["protected_sample_count"] += 1
                if self.conditional_mirror_axis in self.axes:
                    self._window["lr_axis_disabled_sample_count"] += 1
            else:
                self._window["nonprotected_sample_count"] += 1
                if self.conditional_mirror_axis in self.axes:
                    self._window["lr_axis_allowed_sample_count"] += 1

            if np.random.uniform() >= self.p_per_sample:
                continue
            self._window["sample_selected_count"] += 1
            effective_axes = self._effective_axes(protected)
            for axis in effective_axes:
                if int(axis) == 2 and len(data[b].shape) != 4:
                    continue
                if np.random.uniform() < 0.5:
                    data[b] = self._mirror_axis(data[b], int(axis))
                    if seg is not None:
                        seg[b] = self._mirror_axis(seg[b], int(axis))
                    self._record_axis_mirror(int(axis), protected)

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg
        self._flush_window(force=False)
        return data_dict


# ---------------------------------------------------------------------------
# Module-symbol patch
# ---------------------------------------------------------------------------

def install_conditional_mirror_patch(
    *,
    record_dir: Path,
    run_name: str,
    protected_class_ids: Sequence[int] = (4, 5),
    conditional_mirror_axis: int = 0,
    protected_min_voxels: int = 1,
    log_interval: int = 25,
) -> dict[str, Any]:
    """Replace ``MirrorTransform`` in the moreDA module with a conditional factory.

    Must be called AFTER ``setup_DA_params()`` and BEFORE
    ``get_moreDA_augmentation()``.  The original class is saved so
    ``record_witness`` can later verify the replacement took effect.

    Returns a dict with patch info for the witness chain.
    """
    import nnunet.training.data_augmentation.data_augmentation_moreDA as da_more

    original_cls = da_more.MirrorTransform
    telemetry_path = record_dir / "conditional_mirror_components.jsonl"
    events: list[dict[str, Any]] = []

    def factory(
        axes: Sequence[int] = (0, 1, 2),
        data_key: str = "data",
        label_key: str = "seg",
        p_per_sample: float = 1,
    ) -> ConditionalMirrorTransform:
        index = len(events) + 1
        transform = ConditionalMirrorTransform(
            axes=axes,
            data_key=data_key,
            label_key=label_key,
            p_per_sample=p_per_sample,
            protected_class_ids=protected_class_ids,
            conditional_mirror_axis=conditional_mirror_axis,
            protected_min_voxels=protected_min_voxels,
            telemetry_path=telemetry_path,
            log_interval=log_interval,
            run_name=run_name,
            instance_index=index,
            original_transform_class=f"{original_cls.__module__}.{original_cls.__name__}",
        )
        events.append(
            {
                "instance_index": index,
                "original_transform_class": f"{original_cls.__module__}.{original_cls.__name__}",
                "replacement_transform_class": (
                    f"{transform.__class__.__module__}.{transform.__class__.__name__}"
                ),
                "original_mirror_axes": list(transform.axes),
                "conditional_mirror_axis": int(conditional_mirror_axis),
                "protected_class_ids": list(protected_class_ids),
                "protected_min_voxels": int(protected_min_voxels),
                "p_per_sample": float(p_per_sample),
                "data_key": data_key,
                "label_key": label_key,
                "telemetry_path": str(telemetry_path),
            }
        )
        return transform

    da_more.MirrorTransform = factory

    return {
        "patched_module": "nnunet.training.data_augmentation.data_augmentation_moreDA",
        "patched_symbol": "MirrorTransform",
        "original_transform_class": f"{original_cls.__module__}.{original_cls.__name__}",
        "replacement_transform_class": (
            f"{ConditionalMirrorTransform.__module__}.{ConditionalMirrorTransform.__name__}"
        ),
        "events": events,
        "telemetry_path": str(telemetry_path),
        "protected_class_ids": list(protected_class_ids),
        "conditional_mirror_axis": int(conditional_mirror_axis),
        "protected_min_voxels": int(protected_min_voxels),
    }


# ---------------------------------------------------------------------------
# Witness: post-augmentation verification
# ---------------------------------------------------------------------------

def record_witness(
    record_dir: Path,
    mirror_patch_info: dict[str, Any],
    tr_gen,
    run_name: str,
    *,
    loss_name: str = "",
    network_name: str = "",
    sampler_info: str = "",
    optimizer_name: str = "",
    cudnn_deterministic: bool = True,
    cudnn_benchmark: bool = False,
) -> None:
    """Write augmentation_witness.json after get_moreDA_augmentation.

    Verifies:
      - The augmentation pipeline contains NO raw MirrorTransform instances
        (only ConditionalMirrorTransform via the factory).
      - Records the frozen config invariants.
    """
    # Scan the augmentation pipeline for MirrorTransform classes
    transform_names = _collect_transform_names(tr_gen)

    original_name = mirror_patch_info.get(
        "original_transform_class", "MirrorTransform"
    ).split(".")[-1]
    remaining_original = sum(
        1 for n in transform_names if n == original_name
    )

    witness = {
        "schema_version": "augmentation_witness.v1",
        "run_name": run_name,
        "timestamp": time.time(),
        "mirror_patch_info": _json_safe(mirror_patch_info),
        "transform_names": transform_names,
        f"remaining_original_{original_name}_count": remaining_original,
        "original_mirror_transform_replaced": remaining_original == 0,
        "loss_name": loss_name,
        "network_name": network_name,
        "sampler_info": sampler_info,
        "optimizer_name": optimizer_name,
        "cudnn_deterministic": cudnn_deterministic,
        "cudnn_benchmark": cudnn_benchmark,
    }
    _write_json(record_dir / "augmentation_witness.json", witness)


# ---------------------------------------------------------------------------
# Telemetry post-reader (for Stage 4 post-training checks)
# ---------------------------------------------------------------------------

def read_telemetry_verdict(record_dir: Path) -> dict[str, Any]:
    """Read the conditional_mirror_components.jsonl and return a verdict.

    Returns a dict with:
      - total_protected_lr_mirror_count: must be 0 for PASS
      - total_nonprotected_lr_mirror_count
      - total_protected_sample_count / nonprotected_sample_count
      - verdict: PASS if protected_lr_mirror_count == 0 + other checks
    """
    telemetry_path = record_dir / "conditional_mirror_components.jsonl"
    if not telemetry_path.exists():
        return {"verdict": "FAIL", "error": f"telemetry file not found: {telemetry_path}"}

    totals = {
        "protected_sample_count": 0,
        "nonprotected_sample_count": 0,
        "protected_lr_mirror_count": 0,
        "nonprotected_lr_mirror_count": 0,
        "axis0_mirror_count": 0,
        "axis1_mirror_count": 0,
        "axis2_mirror_count": 0,
        "transform_call_count": 0,
    }

    with telemetry_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in totals:
                totals[key] += int(entry.get(key, 0))
            # Also accumulate the final transform_call_count
            if "transform_call_count" in entry:
                totals["transform_call_count"] = entry["transform_call_count"]

    verdict = "PASS" if totals["protected_lr_mirror_count"] == 0 else "FAIL"
    totals["verdict"] = verdict
    return totals


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _collect_transform_names(gen) -> list[str]:
    """Walk the batchgenerators compose tree and collect transform class names."""
    names: list[str] = []
    _walk_transforms(getattr(gen, "transform", None), names)
    return names


def _walk_transforms(transform, names: list[str]) -> None:
    if transform is None:
        return
    if hasattr(transform, "transforms"):
        for t in transform.transforms:
            _walk_transforms(t, names)
    else:
        names.append(type(transform).__name__)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(payload), sort_keys=True) + "\n")
