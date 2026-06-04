#!/bin/bash

# Khai báo mảng 5 con số seed ngẫu nhiên
SEEDS=(42 123 999 2024 777)

# Định nghĩa phương pháp bạn muốn test (ở đây ví dụ là late_avg)
METHOD="late_avg"
VIEWS="front left right"
DATA_DIR="features/stage1_vsl400"
NUM_CLASSES=400

echo "🚀 BẮT ĐẦU CHẠY 5 SEEDS CHO PHƯƠNG PHÁP: $METHOD"
echo "======================================================"

for seed in "${SEEDS[@]}"; do
    echo ">>> Đang huấn luyện với SEED = $seed <<<"
    
    # Đặt tên thư mục output tự động theo cấu trúc: TênThíNghiệm_seed_SốSeed
    OUTPUT_DIR="VSL400_all3_${METHOD}_seed_${seed}"
    
    # Chạy file Python với các tham số tương ứng
    python train_fusion.py \
        --features_dir $DATA_DIR \
        --views $VIEWS \
        --method $METHOD \
        --num_classes $NUM_CLASSES \
        --seed $seed \
        --output_dir $OUTPUT_DIR
        
    echo "✅ Xong seed $seed! Đã lưu tại $OUTPUT_DIR"
    echo "------------------------------------------------------"
done

echo "🎉 HOÀN THÀNH TOÀN BỘ 5 LẦN CHẠY!"