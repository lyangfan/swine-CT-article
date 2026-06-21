#!/usr/bin/env python3
"""Generate the fixed train/val/test split for swine CT article experiments.

Principles (see data/README.md "Data Split" section):
  - 6:2:2 train/val/test, frozen ONCE and reused across all experiments.
  - TB: stratified by breed (4 breeds x 26 each), each breed -> 16 train / 5 val / 5 test.
  - HZAU: random 6:2:2 (all Yorkshire; no breed or batch stratification).
  - Fixed seed = 42. Split is by case_id only -> no feature leakage.
  - test set is frozen (final eval only); val is for model / hyperparameter selection.

Determinism: identical output on every run (sorted input + per-stratum seeded shuffle).

Inputs (relative to this script):
  ../manifests/case_metadata.csv
Outputs (in this script's dir):
  split_manifest.csv    case_id -> split assignment
  split_summary.txt     source x breed x split cross-tabs
"""
import csv
import random
from collections import defaultdict, Counter
from pathlib import Path

SEED = 42
SPLITS_DIR = Path(__file__).resolve().parent
CASE_META = SPLITS_DIR.parent / "manifests" / "case_metadata.csv"
OUT_MANIFEST = SPLITS_DIR / "split_manifest.csv"
OUT_SUMMARY = SPLITS_DIR / "split_summary.txt"

# Fixed per-stratum (train, val, test) sizes.
TB_PER_BREED = (16, 5, 5)   # 4 breeds x 26 = 104
HZAU_TOTAL = (56, 18, 19)   # 93


def assign_stratum(cases, sizes, rng):
    """Shuffle (in place) and slice into train/val/test by sizes. Returns [(row, split)]."""
    assert sum(sizes) == len(cases), f"{sum(sizes)} != {len(cases)}"
    rng.shuffle(cases)
    n_tr, n_va, n_te = sizes
    out = []
    for r in cases[:n_tr]:
        out.append((r, "train"))
    for r in cases[n_tr:n_tr + n_va]:
        out.append((r, "val"))
    for r in cases[n_tr + n_va:]:
        out.append((r, "test"))
    return out


def main() -> None:
    with CASE_META.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_tb_breed = defaultdict(list)
    hzau_rows = []
    for r in rows:
        if r["source"] == "TB":
            by_tb_breed[r["breed_en"]].append(r)
        elif r["source"] == "HZAU":
            hzau_rows.append(r)

    assigned = []

    # TB: per-breed stratified 16/5/5. Independent RNG per breed for auditability.
    for breed in sorted(by_tb_breed):
        cases = sorted(by_tb_breed[breed], key=lambda r: r["case_id"])
        if len(cases) != 26:
            raise SystemExit(f"TB breed {breed}: expected 26, got {len(cases)}")
        assigned.extend(assign_stratum(cases, TB_PER_BREED, random.Random(SEED)))

    # HZAU: random 56/18/19, no stratification.
    hzau_sorted = sorted(hzau_rows, key=lambda r: r["case_id"])
    if len(hzau_sorted) != 93:
        raise SystemExit(f"HZAU: expected 93, got {len(hzau_sorted)}")
    assigned.extend(assign_stratum(hzau_sorted, HZAU_TOTAL, random.Random(SEED)))

    # Stable output order (by case_id) for clean diffs.
    assigned.sort(key=lambda x: x[0]["case_id"])

    fields = ["case_id", "source", "source_detail", "breed_en", "hzau_batch", "split"]
    with OUT_MANIFEST.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r, split in assigned:
            w.writerow({
                "case_id": r["case_id"],
                "source": r["source"],
                "source_detail": r["source_detail"],
                "breed_en": r["breed_en"],
                "hzau_batch": r["hzau_batch"],
                "split": split,
            })

    write_summary(assigned)
    print(f"wrote {OUT_MANIFEST.name} ({len(assigned)} rows) and {OUT_SUMMARY.name}")


def write_summary(assigned) -> None:
    split_counts = Counter(s for (_, s) in assigned)
    src_x_split = Counter((r["source"], s) for (r, s) in assigned)
    tb_breed_x_split = Counter(
        (r["breed_en"], s) for (r, s) in assigned if r["source"] == "TB"
    )
    src_detail_x_split = Counter((r["source_detail"], s) for (r, s) in assigned)

    lines = []
    lines.append("=== split sizes ===")
    for sp in ("train", "val", "test"):
        lines.append(f"  {sp:5s}: {split_counts[sp]}")
    lines.append(f"  total: {sum(split_counts.values())}")

    lines.append("\n=== source x split ===")
    lines.append(f"  {'source':<6s} {'train':>6s} {'val':>5s} {'test':>5s}")
    for src in ("HZAU", "TB"):
        lines.append(
            f"  {src:<6s} {src_x_split[(src,'train')]:>6d} "
            f"{src_x_split[(src,'val')]:>5d} {src_x_split[(src,'test')]:>5d}"
        )

    lines.append("\n=== TB breed x split ===")
    lines.append(f"  {'breed':<10s} {'train':>6s} {'val':>5s} {'test':>5s}")
    for breed in ("Yorkshire", "Landrace", "Pietrain", "Duroc"):
        lines.append(
            f"  {breed:<10s} {tb_breed_x_split[(breed,'train')]:>6d} "
            f"{tb_breed_x_split[(breed,'val')]:>5d} {tb_breed_x_split[(breed,'test')]:>5d}"
        )

    lines.append("\n=== source_detail x split ===")
    lines.append(f"  {'source_detail':<32s} {'train':>6s} {'val':>5s} {'test':>5s}")
    for sd in sorted({r["source_detail"] for (r, _) in assigned}):
        lines.append(
            f"  {sd:<32s} {src_detail_x_split[(sd,'train')]:>6d} "
            f"{src_detail_x_split[(sd,'val')]:>5d} {src_detail_x_split[(sd,'test')]:>5d}"
        )

    # Class-presence coverage (head only in HZAU, testis only in TB)
    lines.append("\n=== class-presence coverage per split ===")
    lines.append("  (head evaluable on HZAU, testis evaluable on TB)")
    lines.append(f"  {'split':<6s} {'head(HZAU)':>11s} {'testis(TB)':>11s}")
    for sp in ("train", "val", "test"):
        lines.append(
            f"  {sp:<6s} {src_x_split[('HZAU',sp)]:>11d} {src_x_split[('TB',sp)]:>11d}"
        )

    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
