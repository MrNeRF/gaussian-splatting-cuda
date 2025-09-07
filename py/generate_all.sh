#!/usr/bin/env bash
# Run tester.py on every top-level directory under ../data.
# Extras:
# - Time each run and collect results.
# - Print an aligned table at the end.
# - Save a TSV with results to SCRIPT_DIR/timings.tsv
#
# 1) For each top-level scene, run tester.py with the given args (timed).
# 2) Post-run per scene: rename any points3D_dense.bin -> points3D.bin.

set -u -o pipefail

# Location of this script and tester.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTER_PY="$SCRIPT_DIR/reconstruct_roma.py"
DATA_DIR="$SCRIPT_DIR/../data/original"
TSV_OUT="$SCRIPT_DIR/timings.tsv"

if [[ ! -f "$TESTER_PY" ]]; then
  echo "ERROR: tester.py not found at: $TESTER_PY" >&2
  exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: Data directory not found at: $DATA_DIR" >&2
  exit 1
fi

echo "=== Running tester.py for each top-level scene under $DATA_DIR ==="

# results arrays
declare -a RES_SCENES=()
declare -a RES_TIMES=()
declare -a RES_STATUS=()

# TSV header
echo -e "scene\tseconds\tstatus" > "$TSV_OUT"

# Iterate only top-level directories (scenes)
while IFS= read -r -d '' scene_dir; do
  scene_name="$(basename "$scene_dir")"
  echo ""
  echo ">>> Scene: $scene_name"

  # Time the python script with /usr/bin/time (outputs seconds with -f '%e')
  echo "[run] python3 reconstruct_roma.py --scene_root \"$scene_dir\" --images_subdir images_4 --roma_model outdoor"

  time_tmp="$(mktemp)"
  # Run and time; capture exit code
  if /usr/bin/time -f '%e' -o "$time_tmp" \
      python3 "$TESTER_PY" --scene_root "$scene_dir" --images_subdir images_4 --roma_model outdoor; then
    status="OK"
    ret=0
  else
    status="FAIL"
    ret=$?
    echo "[warn] tester.py failed for scene: $scene_name (exit $ret; continuing...)" >&2
  fi

  elapsed="$(cat "$time_tmp" || echo "NA")"
  rm -f "$time_tmp"

  # Post-run: rename any points3D_dense.bin to points3D.bin
  echo "[post] Renaming points3D_dense.bin -> points3D.bin in $scene_name"
  while IFS= read -r -d '' f; do
    dst="$(dirname "$f")/points3D.bin"
    echo "[mv]   $f -> $dst"
    mv -v -- "$f" "$dst"
  done < <(find "$scene_dir" -type f -name 'points3D_dense.bin' -print0)

  # Record results
  RES_SCENES+=("$scene_name")
  RES_TIMES+=("$elapsed")
  RES_STATUS+=("$status")

  # Append to TSV (tab-separated)
  echo -e "${scene_name}\t${elapsed}\t${status}" >> "$TSV_OUT"

done < <(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d -print0)

echo ""
echo "=== Timing Summary ==="

# Compute width for scene column
maxlen=5
for s in "${RES_SCENES[@]}"; do
  (( ${#s} > maxlen )) && maxlen=${#s}
done

printf "%-${maxlen}s  %10s  %s\n" "scene" "seconds" "status"
printf "%-${maxlen}s  %10s  %s\n" "$(printf '—%.0s' $(seq 1 $maxlen))" "----------" "------"

for i in "${!RES_SCENES[@]}"; do
  printf "%-${maxlen}s  %10s  %s\n" "${RES_SCENES[$i]}" "${RES_TIMES[$i]}" "${RES_STATUS[$i]}"
done

echo ""
echo "Saved TSV: $TSV_OUT"
echo "=== Done. ==="
