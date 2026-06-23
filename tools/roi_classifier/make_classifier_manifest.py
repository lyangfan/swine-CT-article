#!/usr/bin/env python3
"""make_classifier_manifest.py — §4.2 / Stage 2

Derive the classifier split manifest from the article split_manifest.csv +
GT label voxel scanning.

Output schema (9 columns):
  case_id, split, source, source_detail, breed_en,
  image_path, label_path, head_present, testis_present

Consistency gate (§4.3): GT-derived presence MUST equal source-derived
presence for every case.  Violations → stderr, exit ≠ 0, no output CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Import from common.py (package-relative with standalone-script fallback)
# ---------------------------------------------------------------------------
try:
    from .common import (
        FIRST_ROUND_SPLITS,
        HEAD_LABEL_ID,
        RESERVED_TEST_SPLIT,
        SPEC_PATH,
        TESTIS_LABEL_ID,
        ensure_dirs,
        read_csv,
        write_csv,
    )
except ImportError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parent))
    from common import (  # type: ignore
        FIRST_ROUND_SPLITS,
        HEAD_LABEL_ID,
        RESERVED_TEST_SPLIT,
        SPEC_PATH,
        TESTIS_LABEL_ID,
        ensure_dirs,
        read_csv,
        write_csv,
    )


# ---------------------------------------------------------------------------
# Local constants (not in common.py)
# ---------------------------------------------------------------------------

# All three splits: train + val (first-round) + test (reserved)
ALL_SPLITS: set[str] = FIRST_ROUND_SPLITS | {RESERVED_TEST_SPLIT}

# The 9-column output schema mandated by §4.2
CLASSIFIER_MANIFEST_COLUMNS = [
    "case_id",
    "split",
    "source",
    "source_detail",
    "breed_en",
    "image_path",
    "label_path",
    "head_present",
    "testis_present",
]

# Source → presence mapping (§2 — structural fact)
SOURCE_PRESENCE: Dict[str, Dict[str, int]] = {
    "HZAU": {"head_present": 1, "testis_present": 0},
    "TB":   {"head_present": 0, "testis_present": 1},
}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def derive_presence_from_label(
    label_path: Path,
    head_id: int = HEAD_LABEL_ID,
    testis_id: int = TESTIS_LABEL_ID,
) -> Dict[str, int]:
    """Scan a GT label NIfTI and return {head_present, testis_present}."""
    import SimpleITK as sitk  # lazy — only needed on Huawei

    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    img = sitk.ReadImage(str(label_path))
    arr = sitk.GetArrayFromImage(img)  # (z, y, x) numpy array

    return {
        "head_present":   1 if int((arr == head_id).sum()) > 0 else 0,
        "testis_present": 1 if int((arr == testis_id).sum()) > 0 else 0,
    }


def build_manifest(
    split_manifest_path: Path,
    repo_root: Path,
    head_label_id: int = HEAD_LABEL_ID,
    testis_label_id: int = TESTIS_LABEL_ID,
) -> List[dict]:
    """Build the classifier manifest rows with consistency checking.

    Returns a list of dicts (9 columns) on success.
    Exits with status 1 on any consistency violation.
    """
    # Read split manifest via common.read_csv
    split_rows = read_csv(str(split_manifest_path))

    rows: List[dict] = []
    violations: List[str] = []

    for sr in split_rows:
        case_id = sr["case_id"]
        split = sr["split"]
        if split not in ALL_SPLITS:
            raise ValueError(
                f"Unexpected split value '{split}' for case {case_id}"
            )

        source = sr["source"]

        # Build relative image/label paths (§4.2)
        image_rel = f"data/{split}/images/{case_id}.nii.gz"
        label_rel = f"data/{split}/labels/{case_id}.nii.gz"
        image_path_abs = repo_root / image_rel
        label_path_abs = repo_root / label_rel

        # Derive presence from GT label
        gt_presence = derive_presence_from_label(
            label_path_abs, head_label_id, testis_label_id
        )

        # Source-derived presence
        src_presence = SOURCE_PRESENCE.get(source)
        if src_presence is None:
            raise ValueError(f"Unknown source '{source}' for case {case_id}")

        # Consistency check (§4.3)
        for key in ("head_present", "testis_present"):
            gt_val = gt_presence[key]
            src_val = src_presence[key]
            if gt_val != src_val:
                violations.append(
                    f"CONSISTENCY_VIOLATION: case_id={case_id} "
                    f"source={source} {key}: GT={gt_val} source={src_val}"
                )

        rows.append({
            "case_id":        case_id,
            "split":          split,
            "source":         source,
            "source_detail":  sr.get("source_detail", ""),
            "breed_en":        sr.get("breed_en", ""),
            "image_path":     image_rel,
            "label_path":     label_rel,
            "head_present":   str(gt_presence["head_present"]),
            "testis_present": str(gt_presence["testis_present"]),
        })

    # Gate: fail on any violation (§4.3)
    if violations:
        print("ERROR: Presence consistency check FAILED.", file=sys.stderr)
        for v in violations:
            print(v, file=sys.stderr)
        print(
            f"{len(violations)} violation(s) found. "
            "Manifest will NOT be written. Investigate manually.",
            file=sys.stderr,
        )
        sys.exit(1)

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate classifier_split_manifest.csv (Stage 2)"
    )
    parser.add_argument(
        "--repo-root",
        default=str(SPEC_PATH),
        help=(
            "Article repo root "
            "(default: auto-detected from common.py location)."
        ),
    )
    parser.add_argument(
        "--split-manifest",
        default=None,
        help=(
            "Path to split_manifest.csv "
            "(default: <repo-root>/data/splits/split_manifest.csv)."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output CSV path "
            "(default: <repo-root>/data/manifests/classifier_split_manifest.csv)."
        ),
    )
    parser.add_argument(
        "--head-label-id",
        type=int,
        default=HEAD_LABEL_ID,
        help=f"Label class ID for head (default: {HEAD_LABEL_ID}).",
    )
    parser.add_argument(
        "--testis-label-id",
        type=int,
        default=TESTIS_LABEL_ID,
        help=f"Label class ID for testis (default: {TESTIS_LABEL_ID}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths/existence without writing output.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    split_manifest = (
        Path(args.split_manifest).resolve()
        if args.split_manifest
        else (repo_root / "data" / "splits" / "split_manifest.csv").resolve()
    )
    output_path = (
        Path(args.output).resolve()
        if args.output
        else (repo_root / "data" / "manifests" / "classifier_split_manifest.csv").resolve()
    )

    if not repo_root.is_dir():
        sys.exit(f"ERROR: repo-root does not exist: {repo_root}")
    if not split_manifest.exists():
        sys.exit(f"ERROR: split manifest not found: {split_manifest}")

    print(f"Repo root:      {repo_root}")
    print(f"Split manifest: {split_manifest}")
    print(f"Output:         {output_path}")

    if args.dry_run:
        print("DRY-RUN: paths valid, skipping manifest generation.")
        return

    rows = build_manifest(
        split_manifest_path=split_manifest,
        repo_root=repo_root,
        head_label_id=args.head_label_id,
        testis_label_id=args.testis_label_id,
    )

    # Write via common.write_csv (atomic tmp+rename)
    ensure_dirs(output_path.parent)
    write_csv(output_path, rows, CLASSIFIER_MANIFEST_COLUMNS)

    # Summary stats
    n_cases = len(rows)
    n_head = sum(1 for r in rows if r["head_present"] == "1")
    n_testis = sum(1 for r in rows if r["testis_present"] == "1")
    splits: Dict[str, int] = {}
    for r in rows:
        splits[r["split"]] = splits.get(r["split"], 0) + 1

    print(f"\nDone. {n_cases} cases written to {output_path}")
    print(f"  head_present=1:  {n_head}")
    print(f"  testis_present=1: {n_testis}")
    print(f"  By split: {splits}")
    print("  Consistency check: PASS (0 violations)")


if __name__ == "__main__":
    main()
