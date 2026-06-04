"""
extract_wholebody_vsl400.py

Extract COCO-WholeBody 133-kp từ videos VSL400 → HDF5 format (Optimized).
Adapted từ gen_wholebody_mmwlauslan.py cho cấu trúc VSL400.

Chạy riêng cho từng view:
  python tools/extract_wholebody_vsl400.py --view front --shard 0 --num_shards 2 --device cuda:0
  python tools/extract_wholebody_vsl400.py --view front --shard 1 --num_shards 2 --device cuda:1
  python tools/extract_wholebody_vsl400.py --view front --merge_shards --num_shards 2
"""

import argparse
import os
import h5py
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

# ─── CẤU HÌNH ────────────────────────────────────────────────────
VSL400_ROOT  = '/media/ibmelab/ibme31/vsl400_dataset'
OUTPUT_ROOT  = '/media/ibmelab/ibme31/vsl400_dataset/wholebody_h5'

VIEW_DIR_MAP = {
    'front': 'front_view',
    'left':  'left_view',
    'right': 'right_view',
}
# ─────────────────────────────────────────────────────────────────


def get_all_video_ids(vsl400_root):
    """Lấy tất cả video_id từ 3 CSV splits (train+val+test), không trùng lặp."""
    dfs = []
    for split_csv in ['train_labels_clean.csv', 'val_labels_clean.csv', 'test_labels_clean.csv']:
        csv_path = os.path.join(vsl400_root, split_csv)
        if os.path.exists(csv_path):
            dfs.append(pd.read_csv(csv_path))
    if not dfs:
        return []
    all_df = pd.concat(dfs, ignore_index=True)
    all_df['video_id'] = all_df['sample_id'].apply(lambda x: str(x).zfill(6))
    return sorted(all_df['video_id'].unique().tolist())


def merge_shards(output_h5, num_shards):
    print(f"Merging {num_shards} shards into {output_h5}...")
    os.makedirs(os.path.dirname(output_h5), exist_ok=True)
    with h5py.File(output_h5, 'w') as f_out:
        for s in range(num_shards):
            shard_file = f"{output_h5}.shard{s}.h5"
            if not os.path.exists(shard_file):
                print(f"  Warning: shard not found: {shard_file}")
                continue
            with h5py.File(shard_file, 'r') as f_in:
                for video_id in f_in.keys():
                    if video_id in f_out:
                        continue
                    group_in = f_in[video_id]
                    group_out = f_out.create_group(video_id)
                    for ds_name in group_in.keys():
                        f_out.copy(group_in[ds_name], group_out, name=ds_name)
    print("Merge complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--view', type=str, required=True,
                        choices=['front', 'left', 'right'],
                        help='Which view to process')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size cho MMPose Inferencer giúp tăng tốc GPU')
    parser.add_argument('--merge_shards', action='store_true',
                        help='Merge shards and exit')
    args = parser.parse_args()

    view_dir_name = VIEW_DIR_MAP[args.view]
    video_dir = os.path.join(VSL400_ROOT, 'merged_dataset', view_dir_name)
    output_h5 = os.path.join(OUTPUT_ROOT, args.view, 'all.h5')

    if args.merge_shards:
        merge_shards(output_h5, args.num_shards)
        return

    # Lazy import MMPose
    from mmpose.apis import MMPoseInferencer
    print(f"Loading MMPoseInferencer on {args.device} (Batch size: {args.batch_size})...")
    wholebody_detector = MMPoseInferencer(
        'rtmpose-m_8xb64-270e_coco-wholebody-256x192',
        device=args.device
    )

    all_video_ids = get_all_video_ids(VSL400_ROOT)
    if not all_video_ids:
        print("Error: No video IDs found from CSV files.")
        return
        
    print(f"Total unique videos: {len(all_video_ids)}")

    # Shard data
    shard_ids = all_video_ids[args.shard::args.num_shards]
    print(f"Shard {args.shard}/{args.num_shards}: {len(shard_ids)} videos")

    h5_path = output_h5 if args.num_shards == 1 else f"{output_h5}.shard{args.shard}.h5"
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)

    total_processed = 0
    total_skipped = 0
    total_failed = 0
    empty_frames = 0

    with tqdm(total=len(shard_ids), desc=f"[{args.view}] shard {args.shard}/{args.num_shards}") as pbar:
        for video_id in shard_ids:
            video_name = f"{video_id}.mp4"
            video_path = os.path.join(video_dir, video_name)

            if not os.path.exists(video_path):
                print(f"  WARNING: Not found {video_path}")
                total_failed += 1
                pbar.update(1)
                continue

            # Đọc kiểm tra số lượng khung hình thực tế
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if frame_count <= 0:
                print(f"  WARNING: Video corrupted hoặc trống (0 frames): {video_id}")
                total_failed += 1
                pbar.update(1)
                continue

            # Kiểm tra Resume (Chỉ mở file HDF5 khi cần đọc/ghi để tránh lỗi corrupt file)
            video_exists = False
            if os.path.exists(h5_path):
                with h5py.File(h5_path, 'r') as f_check:
                    if video_id in f_check:
                        existing = f_check[video_id]['wholebody_threshold_02'].shape[0]
                        if existing >= frame_count:
                            video_exists = True
                        else:
                            # Nếu file bị thiếu frame cũ, tí nữa mở bằng 'a' sẽ xóa sau
                            pass

            if video_exists:
                total_skipped += 1
                pbar.update(1)
                continue

            try:
                # 🌟 TỐI ƯU: Truyền batch_size trực tiếp vào MMPoseInferencer
                wholebody_results = wholebody_detector(video_path, batch_size=args.batch_size)

                raw_wb_list = []
                thr_wb_list = []

                for result in wholebody_results:
                    preds = result.get('predictions', [])
                    
                    # Kiểm tra xem có detect được người nào không
                    if not preds or len(preds[0]) == 0:
                        empty_frames += 1
                        empty_pose = np.zeros((133, 3), dtype=np.float32)
                        raw_wb_list.append(empty_pose)
                        thr_wb_list.append(empty_pose)
                        continue

                    # Lấy thông tin keypoints và scores của người đầu tiên phát hiện được
                    # (Hạn chế tối đa việc tạo đi tạo lại mảng tạm)
                    kp = np.array(preds[0][0]['keypoints'], dtype=np.float32)       # (133, 2)
                    scores = np.array(preds[0][0]['keypoint_scores'], dtype=np.float32) # (133,)

                    raw_pose = np.empty((133, 3), dtype=np.float32)
                    raw_pose[:, :2] = kp
                    raw_pose[:, 2] = scores

                    # Lọc threshold nhanh bằng vectorization thay vì copy thủ công
                    thr_pose = np.where(raw_pose[:, 2:3] <= 0.2, 0.0, raw_pose).astype(np.float32)

                    raw_wb_list.append(raw_pose)
                    thr_wb_list.append(thr_pose)

                raw_arr = np.array(raw_wb_list, dtype=np.float32)
                thr_arr = np.array(thr_wb_list, dtype=np.float32)

                if len(raw_arr) > 0:
                    # Mở file để ghi dữ liệu ngay lập tức rồi đóng lại ngay
                    with h5py.File(h5_path, 'a') as f_out:
                        if video_id in f_out:
                            del f_out[video_id]  # Xóa bản ghi lỗi/cũ nếu có trước khi ghi mới
                        
                        grp = f_out.create_group(video_id)
                        grp.create_dataset('raw_wholebody', data=raw_arr,
                                           compression='gzip', compression_opts=4,
                                           chunks=(1, 133, 3))
                        grp.create_dataset('wholebody_threshold_02', data=thr_arr,
                                           compression='gzip', compression_opts=4,
                                           chunks=(1, 133, 3))
                    total_processed += 1
                else:
                    total_failed += 1

            except Exception as e:
                print(f"  WARNING: Exception for {video_id}: {e}")
                total_failed += 1

            pbar.update(1)

    print(f"\n=== [{args.view}] Shard {args.shard} Summary ===")
    print(f"  Processed : {total_processed}")
    print(f"  Skipped   : {total_skipped}")
    print(f"  Failed    : {total_failed}")
    print(f"  Empty frames: {empty_frames}")


if __name__ == '__main__':
    main()