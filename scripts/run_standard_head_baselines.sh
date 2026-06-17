#!/usr/bin/env bash
set -euo pipefail

PROCESSED_DIR="data/processed"
TARGET_COL="log2"

HEADS=(
  "concat"
  "film"
  "query"
)

# Discover single-cell conditions from host embedding directories.
mapfile -t CONDITIONS < <(
  find "${PROCESSED_DIR}/host_emb" \
    -mindepth 1 \
    -maxdepth 1 \
    -type d \
    -printf '%f\n' | sort
)

if [ "${#CONDITIONS[@]}" -eq 0 ]; then
  echo "No conditions discovered under ${PROCESSED_DIR}/host_emb" >&2
  exit 1
fi

echo "Discovered conditions:"
printf '  %s\n' "${CONDITIONS[@]}"
echo

COMMON_ARGS=(
  --processed-dir "${PROCESSED_DIR}"
  --target-col "${TARGET_COL}"
  --standardize-target

  --batch-size 256
  --epochs 100
  --lr 1e-3
  --weight-decay 1e-4
  --loss mse
  --grad-clip-norm 1.0
  --patience 15
  --min-delta 0.0

  --dropout 0.0
  --activation gelu

  --fusion-dim 256

  --query-num-heads 4
  --query-num-sequence-slots 8
  --query-num-queries 4
  --query-pooling mean

  --dna-pooling mean
  --host-pooling mean
  --num-workers 0
  --device auto
  --seed 1337
)

run_one() {
  local head="$1"
  local condition_label="$2"
  shift 2

  local run_ts
  run_ts="$(date +%Y%m%d-%H%M%S)"

  local run_name="${run_ts}_${head}"
  local output_dir="runs/${head}_head/${condition_label}_${TARGET_COL}/${run_name}"

  echo "================================================================"
  echo "Running baseline"
  echo "  head:       ${head}"
  echo "  condition:  ${condition_label}"
  echo "  run_name:   ${run_name}"
  echo "  output_dir: ${output_dir}"
  echo "================================================================"

  python scripts/train_expression_head.py \
    --head "${head}" \
    "${COMMON_ARGS[@]}" \
    --run-name "${run_name}" \
    --output-dir "${output_dir}" \
    "$@"
}

# 1. Multi-condition runs: omit --conditions to include all conditions.
for head in "${HEADS[@]}"; do
  run_one "${head}" "all_conditions"
done

# 2. Single-cell runs: one run per condition per head.
for condition in "${CONDITIONS[@]}"; do
  for head in "${HEADS[@]}"; do
    run_one "${head}" "${condition}" --conditions "${condition}"
  done
done