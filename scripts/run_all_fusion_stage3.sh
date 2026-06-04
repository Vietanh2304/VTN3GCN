#!/bin/bash
# run_all_fusion_stage3.sh — Stage 3 triple fusion: 1 triple x 4 methods = 4 configs.
#
# Usage:
#   bash run_all_fusion_stage3.sh             # run all 4
#   bash run_all_fusion_stage3.sh concat      # run only one method

set -e
set -u

REPO_DIR="/media/ibmelab/ibme31/VTN3GCN_CBAM"
RESULTS_ROOT="${REPO_DIR}/results/Stage3"
DEVICE="cuda:1"
EPOCHS=100
PATIENCE=15

TRIPLE="left center right"      # all 3 views, order = L, C, R
METHODS=(concat addweighted crossattn gmu)

cd "$REPO_DIR"
mkdir -p "$RESULTS_ROOT"

# Allow single-method override: bash run_all_fusion_stage3.sh gmu
if [ $# -eq 1 ]; then
    METHODS=($1)
fi

run_one() {
    local method="$1"
    local out_dir="${RESULTS_ROOT}/LCR_${method}"
    local log_file="${out_dir}/train.log"

    mkdir -p "$out_dir"

    echo ""
    echo "############################################################"
    echo "# [$(date +%H:%M:%S)] Stage 3 | triple=LCR method=$method"
    echo "#   views: $TRIPLE"
    echo "#   out_dir: $out_dir"
    echo "############################################################"

    if [ -f "${out_dir}/result.json" ]; then
        echo "  SKIP: result.json already exists in $out_dir"
        return 0
    fi

    python train_fusion.py \
        --views $TRIPLE \
        --method "$method" \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        --device "$DEVICE" \
        --output_dir "$out_dir" \
        2>&1 | tee "$log_file"
}

T_START=$(date +%s)
for method in "${METHODS[@]}"; do
    run_one "$method"
done
T_END=$(date +%s)

echo ""
echo "############################################################"
echo "# STAGE 3 DONE in $(( (T_END - T_START) / 60 )) min"
echo "############################################################"

python aggregate_fusion_results.py --results_dir "$RESULTS_ROOT" \
    --out "${RESULTS_ROOT}/summary.csv" || true
