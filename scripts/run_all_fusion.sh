#!/bin/bash
# run_all_fusion.sh — Stage 2 pairwise fusion: 3 pairs x 4 methods = 12 configs.
#
# Usage:
#   bash run_all_fusion.sh                # run all 12
#   bash run_all_fusion.sh LC concat      # run only one config
#
# Logs to results/Stage2/<pair>_<method>/train.log

set -e
set -u

REPO_DIR="/media/ibmelab/ibme31/VTN3GCN_CBAM"
RESULTS_ROOT="${REPO_DIR}/results/Stage2"
DEVICE="cuda:1"
EPOCHS=100
PATIENCE=15

# Pair name -> view list (space-separated, matching --views order)
declare -A PAIRS=(
    [LC]="left center"
    [LR]="left right"
    [CR]="center right"
)
PAIR_ORDER=(LC LR CR)
METHODS=(concat addweighted crossattn gmu)

cd "$REPO_DIR"
mkdir -p "$RESULTS_ROOT"

# Allow single-config override: bash run_all_fusion.sh LC concat
if [ $# -eq 2 ]; then
    PAIR_ORDER=($1)
    METHODS=($2)
fi

run_one() {
    local pair="$1"
    local method="$2"
    local views="${PAIRS[$pair]}"
    local out_dir="${RESULTS_ROOT}/${pair}_${method}"
    local log_file="${out_dir}/train.log"

    mkdir -p "$out_dir"

    echo ""
    echo "############################################################"
    echo "# [$(date +%H:%M:%S)] Running: pair=$pair method=$method"
    echo "#   views: $views"
    echo "#   out_dir: $out_dir"
    echo "############################################################"

    if [ -f "${out_dir}/result.json" ]; then
        echo "  SKIP: result.json already exists in $out_dir"
        return 0
    fi

    python train_fusion.py \
        --views $views \
        --method "$method" \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        --device "$DEVICE" \
        --output_dir "$out_dir" \
        2>&1 | tee "$log_file"
}

# Main loop
T_START=$(date +%s)
for pair in "${PAIR_ORDER[@]}"; do
    for method in "${METHODS[@]}"; do
        run_one "$pair" "$method"
    done
done
T_END=$(date +%s)

echo ""
echo "############################################################"
echo "# ALL DONE in $(( (T_END - T_START) / 60 )) min"
echo "############################################################"

# Aggregate
python aggregate_fusion_results.py --results_dir "$RESULTS_ROOT" \
    --out "${RESULTS_ROOT}/summary.csv" || true
