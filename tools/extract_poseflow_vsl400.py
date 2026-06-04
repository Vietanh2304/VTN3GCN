"""
extract_poseflow_vsl400.py

Compute poseflow từ wholebody H5 → poseflow H5 cho VSL400 (Optimized version).
Dùng cùng thuật toán toán học với extract_poseflow_mmwlauslan.py nhưng tối ưu vector hóa.

Chạy:
  python tools/extract_poseflow_vsl400.py --view front
  python tools/extract_poseflow_vsl400.py --view left
  python tools/extract_poseflow_vsl400.py --view right
"""

import argparse
import os
import h5py
import numpy as np
from tqdm.auto import tqdm

# ─────────────────────────────────────────────────────────────────
# CORE ALGORITHM (Đã tối ưu hóa vector bằng NumPy)
# ─────────────────────────────────────────────────────────────────

def calc_pose_flow_vectorized(prev, nxt):
    """Tính toán góc và độ lớn dịch chuyển cho toàn bộ 133 điểm cùng lúc."""
    result = np.zeros_like(prev) # (133, 2)
    
    # Tạo mặt nạ kiểm tra xem điểm có bị trống (all zeros) không
    # True nếu điểm hợp lệ (có dữ liệu)
    valid_prev = np.any(prev != 0, axis=1) 
    valid_nxt = np.any(nxt != 0, axis=1)
    valid_mask = valid_prev & valid_nxt  # (133,)

    if not np.any(valid_mask):
        return result

    # Tính hiệu tọa độ y và x của các điểm hợp lệ
    diff = nxt[valid_mask] - prev[valid_mask]
    
    # Tính toán góc (ang) và độ lớn (mag) bằng vector
    ang = np.arctan2(diff[:, 1], diff[:, 0])
    mag = np.linalg.norm(diff, axis=1)

    # Nạp ngược lại kết quả vào mảng
    result[valid_mask, 0] = ang
    result[valid_mask, 1] = mag
    return result


def impute_missing_keypoints_vectorized(poses):
    """Điền các điểm thiếu (0,0) bằng điểm gần nhất theo trục thời gian (T)."""
    T, V, C = poses.shape
    # Tạo mặt nạ xác định các điểm hợp lệ
    valid_mask = np.any(poses != 0, axis=2) # (T, V)

    for v in range(V):
        valid_indices = np.where(valid_mask[:, v])[0]
        if len(valid_indices) == 0:
            continue # Toàn bộ video không có điểm này thì bỏ qua
            
        missing_indices = np.where(~valid_mask[:, v])[0]
        for t in missing_indices:
            # Tìm index của khung hình hợp lệ có khoảng cách gần t nhất
            closest_idx = valid_indices[np.argmin(np.abs(valid_indices - t))]
            poses[t, v] = poses[closest_idx, v]
    return poses


def normalize_by_neck_dist_vectorized(poses):
    """Chuẩn hóa tọa độ theo khoảng cách từ đỉnh đầu đến cổ."""
    # neck = (poses[:, 5] + poses[:, 6]) / 2.0
    neck = (poses[:, 5, :] + poses[:, 6, :]) / 2.0  # (T, 2)
    head_top = poses[:, 0, :]  # (T, 2)
    
    # Tính neck_length cho toàn bộ khung hình cùng lúc
    neck_lengths = np.linalg.norm(neck - head_top, axis=1, keepdims=True)  # (T, 1)
    
    # Tránh chia cho 0
    valid_mask = neck_lengths[:, 0] >= 1e-6
    
    # Khởi tạo mảng kết quả sao chép từ mảng gốc
    new_poses = poses.copy()
    if np.any(valid_mask):
        # Chia tọa độ các frame hợp lệ cho neck_length tương ứng
        new_poses[valid_mask] /= neck_lengths[valid_mask, :, np.newaxis]
        
    return new_poses

# ─────────────────────────────────────────────────────────────────

INPUT_ROOT  = '/media/ibmelab/ibme31/vsl400_dataset/wholebody_h5'
OUTPUT_ROOT = '/media/ibmelab/ibme31/vsl400_dataset/poseflow_h5'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--view', type=str, required=True,
                        choices=['front', 'left', 'right'])
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    args = parser.parse_args()

    input_h5  = os.path.join(INPUT_ROOT, args.view, 'all.h5')
    output_h5 = os.path.join(OUTPUT_ROOT, args.view, 'all.h5')

    if not os.path.exists(input_h5):
        print(f"ERROR: Input H5 not found: {input_h5}")
        return

    os.makedirs(os.path.dirname(output_h5), exist_ok=True)

    total_processed = 0
    total_skipped = 0
    total_failed = 0

    # Chỉ mở file đọc để lấy danh sách key
    with h5py.File(input_h5, 'r') as f_in:
        all_ids = list(f_in.keys())
    
    shard_ids = all_ids[args.shard::args.num_shards]
    print(f"[{args.view}] {len(shard_ids)} videos to process")

    for video_id in tqdm(shard_ids, desc=f"poseflow [{args.view}]"):
        
        # Kiểm tra trạng thái Resume (Tránh mở file liên tục trong vòng lặp)
        if os.path.exists(output_h5):
            with h5py.File(output_h5, 'r') as f_check:
                if video_id in f_check and 'poseflow' in f_check[video_id]:
                    total_skipped += 1
                    continue

        try:
            # Đọc dữ liệu của video đó
            with h5py.File(input_h5, 'r') as f_in:
                if video_id not in f_in or 'wholebody_threshold_02' not in f_in[video_id]:
                    total_failed += 1
                    continue
                wb_data = f_in[video_id]['wholebody_threshold_02'][:]  # (T, 133, 3)

            T = wb_data.shape[0]
            if T < 2:
                total_failed += 1
                continue

            # 🌟 TỐI ƯU SỐ HỌC: Trích xuất nhanh (T, 133, 2) loại bỏ cột score thứ 3
            poses = wb_data[:, :, :2].astype(np.float32)

            # Thực thi chuỗi thuật toán đã vector hóa
            poses = impute_missing_keypoints_vectorized(poses)
            poses = normalize_by_neck_dist_vectorized(poses)

            # Tính toán dòng chảy (Poseflow)
            flow_list = []
            for i in range(1, T):
                flow = calc_pose_flow_vectorized(poses[i-1], poses[i])
                flow_list.append(flow)

            poseflow_arr = np.array(flow_list, dtype=np.float32)  # (T-1, 133, 2)

            # Lưu dữ liệu an toàn (Mở -> Ghi -> Đóng lập tức)
            if len(poseflow_arr) > 0:
                with h5py.File(output_h5, 'a') as f_out:
                    if video_id in f_out:
                        del f_out[video_id]
                    grp = f_out.create_group(video_id)
                    grp.create_dataset('poseflow', data=poseflow_arr,
                                       compression='gzip', compression_opts=4,
                                       chunks=(1, 133, 2))
                total_processed += 1
            else:
                total_failed += 1

        except Exception as e:
            print(f"  WARNING: {video_id}: {e}")
            total_failed += 1

    print(f"\n=== [{args.view}] Poseflow Summary ===")
    print(f"  Processed : {total_processed}")
    print(f"  Skipped   : {total_skipped}")
    print(f"  Failed    : {total_failed}")


if __name__ == '__main__':
    main()