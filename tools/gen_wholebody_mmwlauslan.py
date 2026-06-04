"""
gen_wholebody_mmwlauslan.py

Extract 133-kp COCO-WholeBody keypoints from videos and output to HDF5 format 
(one .h5 per split) for efficient training. Replaces previous JSON-per-frame logic.
"""

import argparse
import os
import h5py
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import time
import glob

def merge_shards(output_h5, num_shards):
    print(f"Merging {num_shards} shards into {output_h5}...")
    with h5py.File(output_h5, 'w') as f_out:
        for s in range(num_shards):
            shard_file = f"{output_h5}.shard{s}.h5"
            if not os.path.exists(shard_file):
                print(f"Warning: Shard file not found: {shard_file}")
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
    parser = argparse.ArgumentParser(description="Extract wholebody keypoints to HDF5")
    parser.add_argument("--csv_path", type=str, required=True, help="Filtered CSV file")
    parser.add_argument("--video_dir", type=str, required=True, help="Video directory")
    parser.add_argument("--output_h5", type=str, required=True, help="Output HDF5 path")
    parser.add_argument("--shard", type=int, default=0, help="Shard index")
    parser.add_argument("--num_shards", type=int, default=1, help="Total shards")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for MMPose")
    parser.add_argument("--merge_shards", action="store_true", help="Merge shards and exit")
    
    args = parser.parse_args()
    
    if args.merge_shards:
        merge_shards(args.output_h5, args.num_shards)
        return

    # Lazy import to keep --help fast
    from mmpose.apis import MMPoseInferencer
    print(f"Loading MMPoseInferencer on {args.device}...")
    wholebody_detector = MMPoseInferencer("rtmpose-m_8xb64-270e_coco-wholebody-256x192", device=args.device)
    
    df = pd.read_csv(args.csv_path)
    df_shard = df.iloc[args.shard::args.num_shards]
    
    views = ['center', 'left', 'right']
    
    # Collect videos to process
    videos_to_process = []
    for _, row in df_shard.iterrows():
        for view in views:
            if not pd.isna(row[view]):
                videos_to_process.append(row[view])
                
    h5_path = args.output_h5
    if args.num_shards > 1:
        h5_path = f"{args.output_h5}.shard{args.shard}.h5"
        
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    
    total_processed = 0
    total_skipped = 0
    total_failed = 0
    empty_frames = 0
    
    with h5py.File(h5_path, 'a') as f_out:
        with tqdm(total=len(videos_to_process), desc=f"Shard {args.shard}/{args.num_shards}") as pbar:
            for file_name in videos_to_process:
                video_url = os.path.join(args.video_dir, file_name)
                video_id = file_name.replace(".mp4", "")
                
                if not os.path.exists(video_url):
                    print(f"WARNING: Video not found {video_url}")
                    total_failed += 1
                    pbar.update(1)
                    continue
                    
                cap = cv2.VideoCapture(video_url)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                if video_id in f_out:
                    existing_shape = f_out[video_id]['wholebody_threshold_02'].shape[0]
                    if existing_shape >= frame_count and frame_count > 0:
                        total_skipped += 1
                        pbar.update(1)
                        continue
                    else:
                        # Recompute if frame count mismatch
                        del f_out[video_id]
                
                try:
                    wholebody_results = wholebody_detector(video_url)
                    
                    raw_wb_list = []
                    thr_wb_list = []
                    
                    for idx, wholebody_result in enumerate(wholebody_results):
                        preds = wholebody_result.get('predictions', [])
                        if not preds or len(preds[0]) == 0:
                            # Edge case: 0 persons detected
                            empty_frames += 1
                            raw_wb_list.append(np.zeros((133, 3), dtype=np.float32))
                            thr_wb_list.append(np.zeros((133, 3), dtype=np.float32))
                            continue
                            
                        wholebody = np.array(preds[0][0]['keypoints'])
                        prob = np.array(preds[0][0]['keypoint_scores'])
                        
                        assert wholebody.shape == (133, 2), f"Unexpected wholebody shape {wholebody.shape}, expected (133, 2)"
                        assert prob.shape == (133,), f"Unexpected prob shape {prob.shape}, expected (133,)"
                        
                        raw_pose = np.zeros((133, 3), dtype=np.float32)
                        raw_pose[:, :2] = wholebody
                        raw_pose[:, 2] = prob
                        
                        thr_pose = raw_pose.copy()
                        thr_pose[prob <= 0.2] = 0.0
                        
                        raw_wb_list.append(raw_pose)
                        thr_wb_list.append(thr_pose)
                        
                    raw_wb_arr = np.array(raw_wb_list, dtype=np.float32)
                    thr_wb_arr = np.array(thr_wb_list, dtype=np.float32)
                    
                    if len(raw_wb_arr) > 0:
                        grp = f_out.create_group(video_id)
                        grp.create_dataset("raw_wholebody", data=raw_wb_arr, compression="gzip", compression_opts=4, chunks=(1, 133, 3))
                        grp.create_dataset("wholebody_threshold_02", data=thr_wb_arr, compression="gzip", compression_opts=4, chunks=(1, 133, 3))
                        total_processed += 1
                    else:
                        total_failed += 1
                        
                except Exception as e:
                    print(f"WARNING: Exception processing {video_url}: {str(e)}")
                    total_failed += 1
                    
                pbar.update(1)
                
    print(f"\n--- Shard {args.shard} Summary ---")
    print(f"Total processed: {total_processed}")
    print(f"Total skipped (resume): {total_skipped}")
    print(f"Total failed/missing: {total_failed}")
    print(f"Total empty frames (0 persons): {empty_frames}")

if __name__ == "__main__":
    main()
