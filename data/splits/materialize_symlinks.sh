#!/usr/bin/env bash
# Materialize train/val/test symlink directories from split_manifest.csv.
#
# MUST run on Huawei (paca_share) where the real image/label files resolve.
# Creates data/{train,val,test}/{images,labels}/ as symlinks into the real
# labeled_197 data root, exactly mirroring how data/images/ and data/labels/
# are set up. Idempotent: safe to re-run (clears existing symlinks first).
set -euo pipefail

DATA=/home/share/hzau/home/liuyangfan/swine-CT-article/data
REAL=/home/hzau/whcs-share37/liuyangfan/nnunet_medsam_semisup/data/labeled_197
MANIFEST="$DATA/splits/split_manifest.csv"

if [ ! -f "$MANIFEST" ]; then
  echo "ERROR: $MANIFEST not found. Sync splits/ from local first." >&2
  exit 1
fi

for split in train val test; do
  mkdir -p "$DATA/$split/images" "$DATA/$split/labels"
  find "$DATA/$split/images" "$DATA/$split/labels" -type l -delete 2>/dev/null || true
done

while IFS=, read -r case_id source source_detail breed_en hzau_batch split; do
  [ "$case_id" = "case_id" ] && continue  # skip header
  # strip any CR (defensive against CRLF line endings)
  split=${split%$'\r'}
  ln -sf "$REAL/images/$case_id.nii.gz" "$DATA/$split/images/$case_id.nii.gz"
  ln -sf "$REAL/labels/$case_id.nii.gz" "$DATA/$split/labels/$case_id.nii.gz"
done < "$MANIFEST"

echo "=== symlink counts ==="
for split in train val test; do
  n_img=$(find "$DATA/$split/images" -type l | wc -l | tr -d ' ')
  n_lbl=$(find "$DATA/$split/labels" -type l | wc -l | tr -d ' ')
  echo "$split: images=$n_img labels=$n_lbl"
done
