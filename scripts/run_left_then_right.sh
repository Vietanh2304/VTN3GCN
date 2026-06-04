#!/bin/bash
cd /media/ibmelab/ibme31/VTN3GCN_CBAM

echo "============================================"
echo "[$(date)] STARTING LEFT"
echo "============================================"
python main.py --config configs/Stage1_SingleView/Stage1_Left_VSL199.yaml 2>&1 | tee train_left_v4.log
LEFT_EXIT=$?

echo ""
echo "============================================"
echo "[$(date)] LEFT FINISHED (exit code: $LEFT_EXIT)"
echo "============================================"

if [ $LEFT_EXIT -ne 0 ]; then
    echo "Left training failed. Skipping Right."
    exit 1
fi

# Clean pyc giữa 2 run
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

echo ""
echo "============================================"
echo "[$(date)] STARTING RIGHT"
echo "============================================"
python main.py --config configs/Stage1_SingleView/Stage1_Right_VSL199.yaml 2>&1 | tee train_right_v4.log
RIGHT_EXIT=$?

echo ""
echo "============================================"
echo "[$(date)] RIGHT FINISHED (exit code: $RIGHT_EXIT)"
echo "============================================"
echo ""
echo "ALL DONE. Summary:"
echo "  Left exit:  $LEFT_EXIT"
echo "  Right exit: $RIGHT_EXIT"
