#!/usr/bin/env python3
"""Audit canonical orientation and ROI geometry prerequisites for ROI projections.

Auto PASS/FAIL verdict (§5 mandated changes):
- Head centroid x must be in left half (cranial side) for HZAU cases
- Testis centroid x must be in right half (caudal side) for TB cases
- Testis centroid y must be consistent across TB cases (all on same side)
- All checks pass → PASS; any violation → FAIL + block + print violating case_ids

Ported from AutoScientists swct06042040 with §5 mandated changes:
  1. Auto PASS/FAIL verdict (NOT manual preview review)
  2. Read classifier_split_manifest.csv (§4.2) instead of raw split manifest
  3. Use config head_label_id=9, testis_label_id=6
  4. Write orientation-verified credential file on PASS
  5. Keep all centroid data in audit json for reviewability
  6. Accept --repo-root argument (default from common.SPEC_PATH)
  7. Accept --manifest argument (default: repo_root/data/manifests/classifier_split_manifest.csv)
  8. Use "val" not "validation"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# numpy and SimpleITK imported lazily inside helper functions

# Fallback support: allow running as `python <this_file>` without -m
_pkg_dir = str(Path(__file__).resolve().parent)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from common import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    HEAD_LABEL_ID,
    SPEC_PATH,
    TESTIS_LABEL_ID,
    ensure_dirs,
    read_csv,
    stage_report_dir,
    utc_stamp,
    write_json,
)


# ---------------------------------------------------------------------------
# Metadata and centroid helpers
# ---------------------------------------------------------------------------


def _read_metadata(path: Path) -> Dict[str, Any]:
    import SimpleITK as sitk  # lazy — only needed on Huawei

    img = sitk.ReadImage(str(path))
    return {
        "size_xyz": tuple(int(x) for x in img.GetSize()),
        "spacing_xyz": tuple(float(x) for x in img.GetSpacing()),
        "direction": tuple(float(x) for x in img.GetDirection()),
        "origin": tuple(float(x) for x in img.GetOrigin()),
    }


def _centroid_from_label(path: Path, label_value: int) -> Optional[List[float]]:
    import numpy as np
    import SimpleITK as sitk  # lazy — only needed on Huawei

    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))  # (z, y, x)
    pts = np.argwhere(arr == int(label_value))
    if pts.size == 0:
        return None
    return [float(x) for x in pts.mean(axis=0).tolist()]  # [z, y, x]


# ---------------------------------------------------------------------------
# Orientation checks (§5)
# ---------------------------------------------------------------------------


def _check_orientation(
    audit_rows: List[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """Run auto PASS/FAIL checks and return (passed, violations).

    Rules:
      1. Head centroid x (index 2 in z,y,x) must be in left half (cranial side)
         for HZAU cases where head_present=1.
      2. Testis centroid x (index 2 in z,y,x) must be in right half (caudal side)
         for TB cases where testis_present=1.
      3. Testis centroid y (index 1 in z,y,x) must be consistent across all TB
         cases (all on same side of y-axis).
    """
    violations: List[str] = []

    # --- Rule 1: head centroid z in cranial half (z > z_mid) for HZAU ---
    # The 2D projection collapses axis 2 (x = L-R).  The horizontal axis
    # of the projection is the volume's z-axis (cranial-caudal).  After
    # transpose + flip_cranial_to_left, high z (cranial) maps to the left
    # side of the 2D image.  Therefore head — at the cranial end — must
    # have z > z_mid (so that it lands on the left).
    for row in audit_rows:
        source = row.get("source", "")
        head_present = row.get("head_present", "0")
        head_centroid = row.get("head_centroid_zyx")
        size_xyz = row.get("size_xyz")

        if source != "HZAU" or str(head_present) != "1":
            continue

        if head_centroid is None:
            violations.append(
                f"RULE1: case_id={row['case_id']} HZAU head_present=1 but no "
                f"head voxels found in label (centroid is None)"
            )
            continue

        if size_xyz is None:
            violations.append(
                f"RULE1: case_id={row['case_id']} missing size_xyz metadata"
            )
            continue

        # GetSize → (x, y, z);  centroid_zyx → [z, y, x]
        size_z = size_xyz[2]   # depth = cranial-caudal extent
        centroid_z = head_centroid[0]  # z in z,y,x

        if centroid_z <= size_z / 2.0:
            violations.append(
                f"RULE1: case_id={row['case_id']} HZAU head centroid z="
                f"{centroid_z:.1f} NOT in cranial half (size_z={size_z}, "
                f"midpoint={size_z / 2.0:.1f}); expected z > {size_z / 2.0:.1f}"
            )

    # --- Rule 2: testis centroid z in caudal half (z < z_mid) for TB ---
    # Complement of Rule 1: testis at the caudal end → low z → right side
    # after flip.
    for row in audit_rows:
        source = row.get("source", "")
        testis_present = row.get("testis_present", "0")
        testis_centroid = row.get("testis_centroid_zyx")
        size_xyz = row.get("size_xyz")

        if source != "TB" or str(testis_present) != "1":
            continue

        if testis_centroid is None:
            violations.append(
                f"RULE2: case_id={row['case_id']} TB testis_present=1 but no "
                f"testis voxels found in label (centroid is None)"
            )
            continue

        if size_xyz is None:
            violations.append(
                f"RULE2: case_id={row['case_id']} missing size_xyz metadata"
            )
            continue

        size_z = size_xyz[2]
        centroid_z = testis_centroid[0]

        if centroid_z >= size_z / 2.0:
            violations.append(
                f"RULE2: case_id={row['case_id']} TB testis centroid z="
                f"{centroid_z:.1f} NOT in caudal half (size_z={size_z}, "
                f"midpoint={size_z / 2.0:.1f}); expected z < {size_z / 2.0:.1f}"
            )

    # --- Rule 3: testis centroid y consistent across TB cases ---
    tb_testis_sides: List[Tuple[str, float, float, float]] = []  # (case_id, centroid_y, size_y, side)
    for row in audit_rows:
        source = row.get("source", "")
        testis_present = row.get("testis_present", "0")
        testis_centroid = row.get("testis_centroid_zyx")
        size_xyz = row.get("size_xyz")

        if source != "TB" or str(testis_present) != "1":
            continue
        if testis_centroid is None or size_xyz is None:
            continue  # already caught by Rule 2

        size_y = size_xyz[1]
        centroid_y = testis_centroid[1]
        side = "top" if centroid_y < size_y / 2.0 else "bottom"
        tb_testis_sides.append((row["case_id"], centroid_y, float(size_y), side))

    if len(tb_testis_sides) >= 2:
        first_side = tb_testis_sides[0][3]
        for case_id, centroid_y, size_y, side in tb_testis_sides:
            if side != first_side:
                violations.append(
                    f"RULE3: case_id={case_id} TB testis centroid y="
                    f"{centroid_y:.1f} on {side} half (size_y={size_y:.0f}), "
                    f"but other TB cases have testis on {first_side} half; "
                    f"y-side must be consistent across all TB cases"
                )

    passed = len(violations) == 0
    return passed, violations


# ---------------------------------------------------------------------------
# Main audit function
# ---------------------------------------------------------------------------


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else (
        repo_root / "data" / "manifests" / "classifier_split_manifest.csv"
    )
    output_root = Path(args.output_root) if args.output_root else DEFAULT_OUTPUT_ROOT
    stamp = args.stamp or utc_stamp()
    head_label = HEAD_LABEL_ID
    testis_label = TESTIS_LABEL_ID

    if not manifest_path.exists():
        raise FileNotFoundError(f"Classifier split manifest not found: {manifest_path}")

    rows = read_csv(manifest_path)
    if len(rows) != 197:
        print(
            f"WARNING: expected 197 cases, got {len(rows)} — proceeding anyway",
            file=sys.stderr,
        )

    # Validate split values use "val" not "validation"
    unexpected_splits = set(r.get("split", "") for r in rows) - {"train", "val", "test"}
    if unexpected_splits:
        raise ValueError(
            f"Unexpected split values in manifest: {unexpected_splits}. "
            f"Expected only: train, val, test."
        )

    if args.dry_run:
        return {
            "status": "DRY_RUN",
            "message": "ROI orientation audit plan validated",
            "repo_root": str(repo_root),
            "manifest": str(manifest_path),
            "output_root": str(output_root),
            "stamp": stamp,
            "case_count": len(rows),
            "head_label_id": head_label,
            "testis_label_id": testis_label,
            "checks": [
                "head_centroid_x_in_left_half_for_HZAU",
                "testis_centroid_x_in_right_half_for_TB",
                "testis_centroid_y_consistent_across_TB",
            ],
        }

    # ---- Per-case audit ----
    audit_rows: List[Dict[str, Any]] = []
    head_centroids: List[List[float]] = []
    testis_centroids: List[List[float]] = []

    for row in rows:
        case_id = row["case_id"]
        split = row.get("split", "")
        source = row.get("source", "")
        head_present = row.get("head_present", "0")
        testis_present = row.get("testis_present", "0")

        # Resolve label_path relative to repo_root
        label_rel = row.get("label_path", "")
        label_path = repo_root / label_rel if label_rel else None

        record: Dict[str, Any] = {
            "case_id": case_id,
            "split": split,
            "source": source,
            "head_present": head_present,
            "testis_present": testis_present,
            "label_path": str(label_path) if label_path else "",
            "array_axis_order": "SimpleITK z,y,x",
            "projection_axis": "axis2_anatomical_left_right",
            "cranial_to_left": True,
        }

        if label_path and label_path.exists():
            meta = _read_metadata(label_path)
            record.update(meta)

            if str(head_present) == "1":
                centroid = _centroid_from_label(label_path, head_label)
                record["head_centroid_zyx"] = centroid
                if centroid is not None:
                    head_centroids.append(centroid)

            if str(testis_present) == "1":
                centroid = _centroid_from_label(label_path, testis_label)
                record["testis_centroid_zyx"] = centroid
                if centroid is not None:
                    testis_centroids.append(centroid)
        else:
            record["label_missing"] = True
            print(
                f"WARNING: label file not found for case {case_id}: {label_path}",
                file=sys.stderr,
            )

        audit_rows.append(record)

    # ---- Run orientation checks (§5) ----
    passed, violations = _check_orientation(audit_rows)

    # ---- Build summary ----
    summary: Dict[str, Any] = {
        "status": "PASS" if passed else "FAIL",
        "message": (
            "All orientation checks passed."
            if passed
            else f"Orientation check FAILED: {len(violations)} violation(s)"
        ),
        "repo_root": str(repo_root),
        "manifest": str(manifest_path),
        "output_root": str(output_root),
        "stamp": stamp,
        "case_count": len(audit_rows),
        "head_label_id": head_label,
        "testis_label_id": testis_label,
        "aggregate_centroid_counts": {
            "head_present_cases_with_centroid": len(head_centroids),
            "testis_present_cases_with_centroid": len(testis_centroids),
        },
        "audit_rows": audit_rows,
        "checks": {
            "rule1_head_x_left_half_HZAU": "PASS" if not any(
                "RULE1:" in v for v in violations
            ) else "FAIL",
            "rule2_testis_x_right_half_TB": "PASS" if not any(
                "RULE2:" in v for v in violations
            ) else "FAIL",
            "rule3_testis_y_consistent_TB": "PASS" if not any(
                "RULE3:" in v for v in violations
            ) else "FAIL",
        },
        "violations": violations if violations else [],
    }

    # ---- Write audit JSON ----
    rdir = stage_report_dir(output_root, stamp)
    ensure_dirs(rdir)
    audit_json_path = rdir / "roi_orientation_audit.json"
    write_json(audit_json_path, summary)

    # ---- Write orientation-verified credential file on PASS (§5.4) ----
    cred_path = output_root / "orientation_verified.json"
    if passed:
        credential = {
            "orientation_verified": True,
            "verified_at": stamp,
            "audit_report": str(audit_json_path),
            "checks_passed": [
                "head_centroid_x_in_left_half_for_HZAU",
                "testis_centroid_x_in_right_half_for_TB",
                "testis_centroid_y_consistent_across_TB",
            ],
        }
        write_json(cred_path, credential)
        summary["credential_file"] = str(cred_path)
    else:
        # Remove any stale credential
        if cred_path.exists():
            cred_path.unlink()

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=str(SPEC_PATH),
        help="Article repo root (default: auto-detected from common.py location).",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to classifier_split_manifest.csv "
        "(default: <repo-root>/data/manifests/classifier_split_manifest.csv).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override artifact root (default: common.DEFAULT_OUTPUT_ROOT).",
    )
    parser.add_argument(
        "--stamp",
        default=None,
        help="Run stamp; defaults to UTC timestamp.",
    )
    parser.add_argument(
        "--head-label-id",
        type=int,
        default=HEAD_LABEL_ID,
        help=f"GT label id for head centroid (default: {HEAD_LABEL_ID}).",
    )
    parser.add_argument(
        "--testis-label-id",
        type=int,
        default=TESTIS_LABEL_ID,
        help=f"GT label id for testis centroid (default: {TESTIS_LABEL_ID}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan only; do not write outputs.",
    )
    args = parser.parse_args(argv)

    import json as _json

    result = audit(args)
    print(_json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))

    if result["status"] == "FAIL":
        violations = result.get("violations", [])
        print(
            f"\nORIENTATION AUDIT FAILED: {len(violations)} violation(s):",
            file=sys.stderr,
        )
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        return 1

    if result["status"] == "PASS":
        print(
            f"\nORIENTATION AUDIT PASSED. "
            f"Credential written to {result.get('credential_file', '')}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
