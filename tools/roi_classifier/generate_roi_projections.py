#!/usr/bin/env python3
"""Generate train/val (and optionally test) ROI half-projection arrays from CT images only.

Foreground defaults to HU > -800, then 2D fill/cleanup, padded bbox, ROI crop,
and aspect-ratio-preserving resize/pad.

Ported from AutoScientists swct06042040 with §6 mandated changes:
  - --include-test flag (gated, default FORBIDDEN)
  - Allowed splits: without --include-test → train+val; with → train+val+test
  - Uses "val" NOT "validation" (article split convention)
  - Only correct ROI roles (head_cranial_half, testis_caudal_half, testis_caudal_lower_half)
  - Reads classifier_split_manifest.csv (§4.2)
  - Accepts --manifest and --repo-root arguments
  - Projection logic unchanged (foreground, fill, cleanup, bbox, ROI crop, variants, resize)
  - --orientation-verified flag required
  - Preview generation preserved
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    from .common import (
        ENDPOINTS,
        FIRST_ROUND_SPLITS,
        ROI_ROLES_BY_ENDPOINT,
        VARIANTS,
        add_common_args,
        command_result,
        forbid_test_rows,
        load_config,
        manifest_dir,
        preview_dir,
        print_json,
        projection_dir,
        read_csv,
        resolve_output_root,
        stage_report_dir,
        utc_stamp,
        write_csv,
        write_json,
    )
except ImportError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parent))
    from common import (  # type: ignore
        ENDPOINTS,
        FIRST_ROUND_SPLITS,
        ROI_ROLES_BY_ENDPOINT,
        VARIANTS,
        add_common_args,
        command_result,
        forbid_test_rows,
        load_config,
        manifest_dir,
        preview_dir,
        print_json,
        projection_dir,
        read_csv,
        resolve_output_root,
        stage_report_dir,
        utc_stamp,
        write_csv,
        write_json,
    )


def _clip_scale(arr, lo: float, hi: float):
    import numpy as np

    arr = np.clip(arr.astype("float32"), lo, hi)
    return (arr - lo) / max(float(hi) - float(lo), 1e-6)


def _resize_with_padding(img, size: tuple[int, int], padding_value: float = 0.0):
    import numpy as np
    from PIL import Image

    h, w = img.shape[:2]
    target_h, target_w = size
    scale = min(target_h / max(h, 1), target_w / max(w, 1))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    if img.ndim == 2:
        pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype("uint8"))
        resized = pil.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("L", (target_w, target_h), int(round(padding_value * 255)))
        canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
        return np.asarray(canvas).astype("float32") / 255.0
    channels = [_resize_with_padding(img[..., idx], size, padding_value) for idx in range(img.shape[-1])]
    return np.stack(channels, axis=-1)


def _load_ct_volume_sitk(path: str | Path):
    sitk = __import__("SimpleITK")
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return arr, {
        "spacing": tuple(float(x) for x in img.GetSpacing()),
        "direction": tuple(float(x) for x in img.GetDirection()),
        "origin": tuple(float(x) for x in img.GetOrigin()),
        "reader": "SimpleITK",
        "array_axis_order": "z,y,x",
    }


def _canonical_lateral(arr_zy, cfg: dict[str, Any]):
    """Convert SimpleITK z,y projection into image coordinates y,x.

    The output image x-axis is cranial-to-caudal. The existing whole-case
    orientation audit found SimpleITK axis-0 increasing is cranial/superior, so
    the default flips the transposed projection to put cranial on the left.
    R2 must verify this with allowed global orientation evidence before use.
    """

    img = arr_zy.T
    if bool(cfg["projection"].get("flip_cranial_to_left", True)):
        img = img[:, ::-1]
    return img


def _project_channels(volume, cfg: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    proj_cfg = cfg["projection"]
    vol = np.asarray(volume).astype("float32")
    if vol.ndim != 3:
        raise ValueError(f"expected 3D CT volume, got shape {vol.shape}")
    muscle_lo, muscle_hi = proj_cfg["channels"]["muscle_window"]["clip_hu"]
    bone_thr = float(proj_cfg["channels"]["bone_only"]["threshold_hu"])
    fg_thr = float(proj_cfg["foreground"]["threshold_hu"])

    muscle = _clip_scale(vol, float(muscle_lo), float(muscle_hi))
    bone_only = np.where(vol > bone_thr, vol, bone_thr)
    bone_only = _clip_scale(bone_only, bone_thr, max(float(vol.max()), bone_thr + 1.0))
    foreground = (vol > fg_thr).astype("float32")

    channels = {
        "muscle_window_mean": muscle.mean(axis=2),
        "muscle_window_p90": np.percentile(muscle, 90, axis=2),
        "foreground_thickness": foreground.mean(axis=2),
        "foreground_occupancy": foreground.max(axis=2),
        "bone_only_mip": bone_only.max(axis=2),
    }
    return {name: _canonical_lateral(value, cfg) for name, value in channels.items()}


def _fill_and_cleanup_foreground(fg2d, cfg: dict[str, Any]):
    import numpy as np

    ndi = __import__("scipy.ndimage", fromlist=["binary_fill_holes", "label"])
    mask = np.asarray(fg2d > 0, dtype=bool)
    if bool(cfg["projection"]["foreground"].get("hole_fill_2d", True)):
        mask = ndi.binary_fill_holes(mask)
    if bool(cfg["projection"]["foreground"].get("connected_component_cleanup_2d", True)):
        labels, count = ndi.label(mask)
        if count > 0:
            areas = np.bincount(labels.ravel())
            areas[0] = 0
            keep_min = max(1, int(round(float(cfg["projection"]["foreground"]["min_component_area_ratio"]) * mask.size)))
            keep = np.where(areas >= keep_min)[0]
            if len(keep) == 0:
                keep = [int(areas.argmax())]
            mask = np.isin(labels, keep)
    return mask


def _bbox_from_mask(mask, padding_ratio: float) -> tuple[int, int, int, int]:
    import numpy as np

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("foreground mask is empty; cannot build ROI bbox")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    h, w = mask.shape
    pad_y = int(round((y1 - y0) * padding_ratio))
    pad_x = int(round((x1 - x0) * padding_ratio))
    return max(0, y0 - pad_y), min(h, y1 + pad_y), max(0, x0 - pad_x), min(w, x1 + pad_x)


def _roi_bbox(bbox: tuple[int, int, int, int], role: str) -> tuple[int, int, int, int]:
    """Compute ROI bounding-box from the padded foreground bbox.

    Only correct ROI roles are supported (§6 change 4):
      - head_cranial_half: cranial 50% of bbox
      - testis_caudal_half: caudal 50% of bbox
      - testis_caudal_lower_half: lower 50% inside caudal 50% of bbox
    """
    y0, y1, x0, x1 = bbox
    mid_x = x0 + max(1, (x1 - x0) // 2)
    if role == "head_cranial_half":
        return y0, y1, x0, mid_x
    if role == "testis_caudal_half":
        return y0, y1, mid_x, x1
    if role == "testis_caudal_lower_half":
        mid_y = y0 + max(1, (y1 - y0) // 2)
        return mid_y, y1, mid_x, x1
    raise ValueError(f"unknown ROI role: {role}")


def _variant_metadata(variant: str) -> dict[str, str]:
    mapping = {
        "muscle_only_mean": ("muscle_window", "mean", "single_channel_replicated"),
        "muscle_only_p90": ("muscle_window", "p90", "single_channel_replicated"),
        "foreground_thickness": ("foreground_thickness", "foreground_thickness", "single_channel_replicated"),
        "bone_only_mip": ("bone_only", "mip_max", "single_channel_replicated"),
        "multi_channel_compact": ("compact", "p90+foreground_thickness+mip_max", "three_channel_stack"),
    }
    channel_family, operator, precursor = mapping[variant]
    return {
        "channel_family": channel_family,
        "projection_operator": operator,
        "normalization_precursor": precursor,
    }


def _variant_array(channels: dict[str, Any], variant: str, roi_bbox: tuple[int, int, int, int], size: tuple[int, int], padding: float):
    import numpy as np

    y0, y1, x0, x1 = roi_bbox

    def crop(name: str):
        return channels[name][y0:y1, x0:x1]

    if variant == "muscle_only_mean":
        arr = _resize_with_padding(crop("muscle_window_mean"), size, padding)
        return np.stack([arr, arr, arr], axis=-1)
    if variant == "muscle_only_p90":
        arr = _resize_with_padding(crop("muscle_window_p90"), size, padding)
        return np.stack([arr, arr, arr], axis=-1)
    if variant == "foreground_thickness":
        arr = _resize_with_padding(crop("foreground_thickness"), size, padding)
        return np.stack([arr, arr, arr], axis=-1)
    if variant == "bone_only_mip":
        arr = _resize_with_padding(crop("bone_only_mip"), size, padding)
        return np.stack([arr, arr, arr], axis=-1)
    if variant == "multi_channel_compact":
        return np.stack(
            [
                _resize_with_padding(crop("muscle_window_p90"), size, padding),
                _resize_with_padding(crop("foreground_thickness"), size, padding),
                _resize_with_padding(crop("bone_only_mip"), size, padding),
            ],
            axis=-1,
        )
    raise ValueError(f"unknown projection variant: {variant}")


def _save_preview(path: Path, array) -> None:
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = array if array.ndim == 3 else np.stack([array, array, array], axis=-1)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype("uint8")).save(path)


def _save_overlay(path: Path, base, bbox: tuple[int, int, int, int], roi: tuple[int, int, int, int]) -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    base_u8 = (np.clip(base, 0, 1) * 255).astype("uint8")
    rgb = np.stack([base_u8, base_u8, base_u8], axis=-1)
    im = Image.fromarray(rgb)
    draw = ImageDraw.Draw(im)
    y0, y1, x0, x1 = bbox
    draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(0, 255, 0), width=2)
    ry0, ry1, rx0, rx1 = roi
    draw.rectangle([rx0, ry0, rx1 - 1, ry1 - 1], outline=(255, 0, 0), width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def _parse_csv_list(value: str | None, default: tuple[str, ...]) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()] if value else list(default)


def _resolve_image_path(image_path_value: str, repo_root: Path) -> Path:
    """Resolve an image_path from the classifier manifest to an absolute path.

    The classifier_split_manifest.csv stores relative paths (e.g.
    data/train/images/case_001.nii.gz).  Resolve them against repo_root.
    If the path is already absolute, return it as-is.
    """
    p = Path(image_path_value)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def generate(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    cfg = load_config(args.config)
    output_root = resolve_output_root(cfg, args.output_root)
    repo_root = Path(args.repo_root).resolve()
    if not repo_root.is_dir():
        raise ValueError(f"repo-root is not a directory: {repo_root}")
    stamp = args.stamp or utc_stamp()

    # ---- Split policy: gated by --include-test (§6 changes 1-2) ----
    if args.include_test:
        allowed_splits = {"train", "val", "test"}
    else:
        allowed_splits = {"train", "val"}

    # ---- Read classifier split manifest (§6 change 5) ----
    manifest_rows_all = read_csv(args.manifest)
    if not manifest_rows_all:
        raise ValueError(f"classifier split manifest is empty: {args.manifest}")

    # Filter to allowed splits (article convention uses "val" not "validation")
    rows = [r for r in manifest_rows_all if r.get("split") in allowed_splits]

    # Gate: if --include-test is NOT set, no test rows may be consumed
    if not args.include_test:
        forbid_test_rows(rows, "ROI projection split rows")

    # ---- Variants, endpoints, ROI roles ----
    variants = _parse_csv_list(args.variants, VARIANTS)
    bad_variants = sorted(set(variants) - set(VARIANTS))
    if bad_variants:
        raise ValueError(f"unknown variants: {bad_variants}")

    endpoint_filter = set(_parse_csv_list(args.endpoints, ENDPOINTS))
    bad_endpoints = sorted(endpoint_filter - set(ROI_ROLES_BY_ENDPOINT))
    if bad_endpoints:
        raise ValueError(f"unknown endpoints: {bad_endpoints}")

    # Default to correct ROI roles only (§6 change 4)
    roi_roles = _parse_csv_list(
        args.roi_roles,
        tuple(r for ep in endpoint_filter for r in ROI_ROLES_BY_ENDPOINT[ep]),
    )
    allowed_roles = {r for ep in endpoint_filter for r in ROI_ROLES_BY_ENDPOINT[ep]}
    bad_roles = sorted(set(roi_roles) - allowed_roles)
    if bad_roles:
        raise ValueError(f"ROI roles not valid for selected endpoints: {bad_roles}")

    # ---- Orientation verification required (§6 change 9) ----
    if not args.orientation_verified:
        raise ValueError(
            "ROI projection requires --orientation-verified after R2 audit"
        )
    if any(
        role.endswith("_lower_half") or role.endswith("_upper_half")
        for role in roi_roles
    ) and not args.lower_direction_verified:
        raise ValueError(
            "lower/upper ROI roles require --lower-direction-verified"
        )

    size = tuple(int(x) for x in cfg["projection"]["input_size"])
    plan = command_result(
        "DRY_RUN" if args.dry_run else "WRITE",
        "ROI projection plan validated",
        output_root=str(output_root),
        stamp=stamp,
        case_count=len(rows),
        allowed_splits=sorted(allowed_splits),
        endpoints=sorted(endpoint_filter),
        roi_roles=roi_roles,
        variants=variants,
        expected_rows=len(rows) * len(roi_roles) * len(variants),
        manifest_path=str(
            manifest_dir(output_root) / f"roi_projection_manifest_{stamp}.csv"
        ),
        test_consumed=args.include_test,
        uses_gt_for_inputs=False,
    )
    if args.dry_run:
        return plan

    proj_root = projection_dir(output_root, stamp)
    prev_root = preview_dir(output_root, stamp)
    padding = float(cfg["projection"].get("padding_value", 0.0))
    manifest_rows: list[dict[str, Any]] = []
    preview_lines = [
        "# ROI Projection Preview Index",
        "",
        f"Stamp: `{stamp}`",
        "",
        "| case_id | split | endpoint | roi_role | variant | preview | overlay |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        case_id = row["case_id"]
        # Resolve image_path: classifier manifest uses relative paths (§6)
        image_path_abs = _resolve_image_path(row["image_path"], repo_root)
        volume, meta = _load_ct_volume_sitk(image_path_abs)
        channels = _project_channels(volume, cfg)
        filled_fg = _fill_and_cleanup_foreground(channels["foreground_occupancy"], cfg)
        bbox = _bbox_from_mask(
            filled_fg, float(cfg["projection"]["foreground"]["bbox_padding_ratio"])
        )
        bbox_y0, bbox_y1, bbox_x0, bbox_x1 = bbox
        bbox_area = max(1, (bbox_y1 - bbox_y0) * (bbox_x1 - bbox_x0))
        for endpoint in sorted(endpoint_filter):
            for role in ROI_ROLES_BY_ENDPOINT[endpoint]:
                if role not in roi_roles:
                    continue
                roi = _roi_bbox(bbox, role)
                ry0, ry1, rx0, rx1 = roi
                roi_area = max(1, (ry1 - ry0) * (rx1 - rx0))
                for variant in variants:
                    arr = _variant_array(channels, variant, roi, size, padding)
                    arr_path = (
                        proj_root / endpoint / role / variant / f"{case_id}.npy"
                    )
                    png_path = (
                        prev_root / endpoint / role / variant / f"{case_id}.png"
                    )
                    overlay_path = (
                        prev_root
                        / endpoint
                        / role
                        / variant
                        / f"{case_id}_overlay.png"
                    )
                    arr_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(arr_path, arr.astype("float32"))
                    _save_preview(png_path, arr)
                    _save_overlay(
                        overlay_path, channels["muscle_window_p90"], bbox, roi
                    )
                    vmeta = _variant_metadata(variant)
                    manifest_rows.append(
                        {
                            "case_id": case_id,
                            "split": row["split"],
                            "endpoint": endpoint,
                            "label": row.get(endpoint, ""),
                            "roi_role": role,
                            "variant": variant,
                            "artifact_role": "training_variant",
                            "array_path": str(arr_path),
                            "preview_path": str(png_path),
                            "overlay_path": str(overlay_path),
                            "image_path": str(image_path_abs),
                            "source_or_cohort": row.get(
                                "source_or_cohort", row.get("source", "")
                            ),
                            "image_shape": getattr(volume, "shape", ""),
                            "spacing": meta.get("spacing"),
                            "reader": meta.get("reader"),
                            "array_axis_order": meta.get("array_axis_order"),
                            "projection_axis": "SimpleITK array axis 2 / anatomical left-right",
                            "canonical_x_axis": "cranial_to_caudal",
                            "canonical_y_axis": "table_back_or_dorsal_ventral_verified",
                            "foreground_threshold_hu": cfg["projection"][
                                "foreground"
                            ]["threshold_hu"],
                            "foreground_fill": "scipy.ndimage.binary_fill_holes_2d",
                            "foreground_cleanup": "scipy.ndimage.label_remove_small_components_2d",
                            "min_component_area_ratio": cfg["projection"][
                                "foreground"
                            ]["min_component_area_ratio"],
                            "bbox_padding_ratio": cfg["projection"]["foreground"][
                                "bbox_padding_ratio"
                            ],
                            "bbox_y0": bbox_y0,
                            "bbox_y1": bbox_y1,
                            "bbox_x0": bbox_x0,
                            "bbox_x1": bbox_x1,
                            "bbox_height": bbox_y1 - bbox_y0,
                            "bbox_width": bbox_x1 - bbox_x0,
                            "bbox_area": bbox_area,
                            "bbox_aspect": (bbox_x1 - bbox_x0)
                            / max(bbox_y1 - bbox_y0, 1),
                            "bbox_foreground_occupancy": float(
                                filled_fg[
                                    bbox_y0:bbox_y1, bbox_x0:bbox_x1
                                ].mean()
                            ),
                            "roi_y0": ry0,
                            "roi_y1": ry1,
                            "roi_x0": rx0,
                            "roi_x1": rx1,
                            "roi_height": ry1 - ry0,
                            "roi_width": rx1 - rx0,
                            "roi_area": roi_area,
                            "roi_aspect": (rx1 - rx0) / max(ry1 - ry0, 1),
                            "roi_bbox_area_fraction": roi_area
                            / max(bbox_area, 1),
                            "channel_family": vmeta["channel_family"],
                            "projection_operator": vmeta["projection_operator"],
                            "normalization_precursor": vmeta[
                                "normalization_precursor"
                            ],
                            "projection_size": f"{size[0]}x{size[1]}",
                            "uses_gt_input": 0,
                            "test_consumed": 1 if args.include_test else 0,
                        }
                    )
                    preview_lines.append(
                        f"| {case_id} | {row['split']} | {endpoint} | {role} | {variant} | `{png_path}` | `{overlay_path}` |"
                    )

    fields = [
        "case_id",
        "split",
        "endpoint",
        "label",
        "roi_role",
        "variant",
        "artifact_role",
        "array_path",
        "preview_path",
        "overlay_path",
        "image_path",
        "source_or_cohort",
        "image_shape",
        "spacing",
        "reader",
        "array_axis_order",
        "projection_axis",
        "canonical_x_axis",
        "canonical_y_axis",
        "foreground_threshold_hu",
        "foreground_fill",
        "foreground_cleanup",
        "min_component_area_ratio",
        "bbox_padding_ratio",
        "bbox_y0",
        "bbox_y1",
        "bbox_x0",
        "bbox_x1",
        "bbox_height",
        "bbox_width",
        "bbox_area",
        "bbox_aspect",
        "bbox_foreground_occupancy",
        "roi_y0",
        "roi_y1",
        "roi_x0",
        "roi_x1",
        "roi_height",
        "roi_width",
        "roi_area",
        "roi_aspect",
        "roi_bbox_area_fraction",
        "channel_family",
        "projection_operator",
        "normalization_precursor",
        "projection_size",
        "uses_gt_input",
        "test_consumed",
    ]
    manifest_path = manifest_dir(output_root) / f"roi_projection_manifest_{stamp}.csv"
    write_csv(manifest_path, manifest_rows, fields)
    rdir = stage_report_dir(output_root, stamp)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "roi_projection_preview_index.md").write_text(
        "\n".join(preview_lines) + "\n", encoding="utf-8"
    )
    write_json(
        rdir / "roi_projection_summary.json",
        {**plan, "status": "PASS", "projection_rows": len(manifest_rows)},
    )
    return {
        **plan,
        "status": "PASS",
        "projection_rows": len(manifest_rows),
        "manifest_path": str(manifest_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)

    # ---- §6 changed arguments ----
    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "Path to classifier_split_manifest.csv (§4.2). "
            "Default: <repo-root>/data/manifests/classifier_split_manifest.csv"
        ),
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help=(
            "Article repo root. Used to resolve relative image_paths from the "
            "classifier manifest and to compute default manifest path. "
            "Default: auto-detected from common.py package location."
        ),
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        default=False,
        help=(
            "If set, generate projections for test split too (train+val+test). "
            "Default: test is FORBIDDEN — only train and val are processed."
        ),
    )

    # ---- Orientation verification (required) ----
    parser.add_argument(
        "--orientation-verified",
        action="store_true",
        help="Required after R2 orientation audit (§6 change 9)",
    )
    parser.add_argument(
        "--lower-direction-verified",
        action="store_true",
        help="Required for lower/upper testis ROI roles",
    )

    # ---- Variant / endpoint / role filtering (unchanged from source) ----
    parser.add_argument(
        "--variants",
        default=None,
        help="Comma-separated variants; default all valid variants",
    )
    parser.add_argument(
        "--endpoints",
        default=None,
        help="Comma-separated endpoints; default head_present,testis_present",
    )
    parser.add_argument(
        "--roi-roles",
        default=None,
        help="Comma-separated ROI roles; default all correct roles for selected endpoints",
    )

    args = parser.parse_args(argv)

    # ---- Resolve defaults that depend on repo-root ----
    try:
        from .common import SPEC_PATH as _AUTO_REPO_ROOT
    except ImportError:
        from common import SPEC_PATH as _AUTO_REPO_ROOT  # type: ignore

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(_AUTO_REPO_ROOT).resolve()
    args.repo_root = str(repo_root)

    if args.manifest is None:
        args.manifest = str(repo_root / "data" / "manifests" / "classifier_split_manifest.csv")

    print_json(generate(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
