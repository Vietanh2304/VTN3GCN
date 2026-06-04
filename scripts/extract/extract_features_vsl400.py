"""
extract_features_vsl400.py

Extract features từ 3 Stage1 checkpoints cho VSL400 (Version Tối Ưu Tốc Độ & Bộ Nhớ).
Output: features_vsl400/{train,val,test}.pt
"""

import os, sys
sys.path.insert(0, '/media/ibmelab/ibme31/VTN3GCN_CBAM')
os.chdir('/media/ibmelab/ibme31/VTN3GCN_CBAM')

import torch
import pandas as pd
import yaml
from tqdm import tqdm
from modelling.vtn_att_poseflow_model import VTNHCPF, VTNHCPF_GCN
from dataset.dataloader import build_dataloader

# Tắt hoàn toàn việc tính toán Gradient trên toàn hệ thống để tiết kiệm tối đa VRAM
torch.set_grad_enabled(False)

DEVICE      = 'cuda:1'
BATCH_SIZE  = 16
NUM_WORKERS = 6
OUTPUT_DIR  = 'features/stage1_vsl400'
SPLITS      = ['train', 'val', 'test']

VSL400_LABEL_ROOT = '/media/ibmelab/ibme31/vsl400_dataset/labels'
TMP_CSV           = '/tmp/extract_vsl400_csv'

VIEW_CONFIGS = {
    'front': {
        'cfg_path':  '/media/ibmelab/ibme31/VTN3GCN_CBAM/configs/Stage1_SingleView/Stage1_Center_VSL400.yaml',
        'ckpt_path': 'checkpoints/vtn_att_poseflow/Stage1 Front VSL400/best_checkpoints.pth',
        'model_cls': VTNHCPF,
    },
    'left': {
        'cfg_path':  '/media/ibmelab/ibme31/VTN3GCN_CBAM/configs/Stage1_SingleView/Stage1_Left_VSL400.yaml',
        'ckpt_path': 'checkpoints/VTNGCN/Stage1 Left VSL400/best_checkpoints.pth',
        'model_cls': VTNHCPF_GCN,
    },
    'right': {
        'cfg_path':  'configs/Stage1_SingleView/Stage1_Right_VSL400.yaml',
        'ckpt_path': 'checkpoints/VTNGCN/Stage1 Right VSL400/best_checkpoints.pth',
        'model_cls': VTNHCPF_GCN,
    },
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Generate per-view CSV từ labels VSL400
print("=== Step 1: Per-view CSV ===")
for split in SPLITS:
    for view, conf in VIEW_CONFIGS.items():
        src_csv = os.path.join(VSL400_LABEL_ROOT, view, f'{split}_labels.csv')
        dst_dir = os.path.join(TMP_CSV, view)
        os.makedirs(dst_dir, exist_ok=True)
        
        if os.path.exists(src_csv):
            df = pd.read_csv(src_csv)
            df = df.rename(columns={'label_id': 'label'})
            dst_csv = os.path.join(dst_dir, f'{split}_labels.csv')
            df.to_csv(dst_csv, index=False)
    print(f"  {split}: CSVs created")

# Khởi tạo kết quả tổng hợp
results = {split: {'front': None, 'left': None, 'right': None, 'labels': None} for split in SPLITS}

# Tải lại dữ liệu cũ nếu script từng bị ngắt quãng (Cơ chế Resume một phần)
for split in SPLITS:
    tmp_path = os.path.join(OUTPUT_DIR, f'partial_{split}.pt')
    if os.path.exists(tmp_path):
        try:
            results[split] = torch.load(tmp_path, map_location='cpu')
            print(f"  Loaded partially extracted features for split: {split}")
        except Exception:
            pass

# Step 2: Extract per view
for view, conf in VIEW_CONFIGS.items():
    print(f"\n=== Step 2: Extract [{view}] ===")

    # Kiểm tra xem view này ở tất cả các split đã được giải quyết ở lần chạy trước chưa
    if all(results[split][view] is not None for split in SPLITS):
        print(f"  [Skip] View {view} đã được trích xuất hoàn chỉnh trước đó.")
        continue

    if not os.path.exists(conf['ckpt_path']):
        print(f"  ERROR: Checkpoint not found: {conf['ckpt_path']}")
        continue

    cfg = yaml.safe_load(open(conf['cfg_path']))
    cfg['training']['pretrained'] = False
    cfg['training']['device'] = DEVICE
    cfg['training']['batch_size'] = BATCH_SIZE
    cfg['training']['num_workers'] = NUM_WORKERS
    cfg['training']['prefetch_factor'] = 2 # Giúp CPU chuẩn bị trước dữ liệu cho GPU

    # Khởi tạo mô hình gọn gàng
    model = conf['model_cls'](**cfg['model'], sequence_length=cfg['data']['num_output_frames'])
    state_dict = torch.load(conf['ckpt_path'], map_location='cpu')
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded: missing={len(missing)}, unexpected={len(unexpected)}")

    model = model.to(DEVICE).eval()

    for split in SPLITS:
        # Nếu split của view này đã có dữ liệu từ trước, bỏ qua không trích xuất lại
        if results[split][view] is not None:
            print(f"    Split {split} của {view} đã xong. Bỏ qua.")
            continue

        csv_file = os.path.join(TMP_CSV, view, f'{split}_labels.csv')
        if not os.path.exists(csv_file):
            continue
            
        sub_df = pd.read_csv(csv_file)
        loader = build_dataloader(cfg, split, is_train=False, model=model, labels=sub_df)

        feats_list, labels_list = [], []
        
        for batch in tqdm(loader, desc=f'  {view} {split}'):
            features, labels = batch[0], batch[1]
            
            # Đẩy nhanh đặc trưng lên GPU bằng cách gom gọn cấu trúc dictionary
            features = {k: v.to(DEVICE, non_blocking=True) for k, v in features.items() if hasattr(v, 'to')}

            clip     = features['clip']
            poseflow = features['poseflow']
            b, t, x, c, h, w = clip.size()
            
            # Trích xuất đặc trưng không gian (RGB)
            z_rgb = model.feature_extractor(clip.view(b, t * x, c, h, w)).view(b, t, -1)

            if view == 'front':
                feat_seq = model.forward_features(features=z_rgb, poseflow=poseflow)
            else:
                keypoints = features['keypoints']
                z_kp = model.feature_extractor_gcn(keypoints)
                feat_seq = model.forward_features(features=z_rgb, poseflow=poseflow, features_keypoint=z_kp)

            # Pooling chiều thời gian và chuyển ngay về CPU để giải phóng VRAM GPU ngay lập tức
            feat = feat_seq.mean(1).cpu()

            feats_list.append(feat)
            labels_list.append(labels)

        if len(feats_list) == 0:
            continue

        feats_all  = torch.cat(feats_list, dim=0)
        labels_all = torch.cat(labels_list, dim=0)
        results[split][view] = feats_all

        # Kiểm tra tính đồng bộ nhãn (Labels) giữa các góc nhìn
        if results[split]['labels'] is None:
            results[split]['labels'] = labels_all
        else:
            if not torch.equal(results[split]['labels'], labels_all):
                print(f"  [Warning] Labels lệch giữa các view ở {split}! Đang đồng bộ bằng cách lấy nhãn hiện tại.")
                results[split]['labels'] = labels_all

        print(f"    -> features {feats_all.shape}, labels {results[split]['labels'].shape}")
        
        # Lưu file tạm sau khi chạy xong mỗi split để phòng hờ rủi ro mất điện
        torch.save(results[split], os.path.join(OUTPUT_DIR, f'partial_{split}.pt'))

    # Dọn dẹp GPU triệt để trước khi đổi sang góc nhìn tiếp theo
    del model
    torch.cuda.empty_cache()

# Step 3: Lưu file cấu trúc cuối cùng sạch sẽ
print("\n=== Step 3: Save final files ===")
for split in SPLITS:
    out_path = os.path.join(OUTPUT_DIR, f'{split}.pt')
    
    if any(results[split][v] is None for v in ['front', 'left', 'right']):
        print(f"  Không thể xuất file {split}.pt cuối cùng vì thiếu dữ liệu của một số góc nhìn.")
        continue
        
    # Tạo bản dict sạch để lưu
    final_dict = {
        'front': results[split]['front'],
        'left': results[split]['left'],
        'right': results[split]['right'],
        'labels': results[split]['labels']
    }
    
    torch.save(final_dict, out_path)
    
    # Xóa file tạm partial
    tmp_path = os.path.join(OUTPUT_DIR, f'partial_{split}.pt')
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
        
    sz = os.path.getsize(out_path) / 1024 / 1024
    print(f"  {out_path}: front {final_dict['front'].shape}, "
          f"left {final_dict['left'].shape}, "
          f"right {final_dict['right'].shape} ({sz:.1f} MB)")

print("\nDone! Toàn bộ đặc trưng đã sẵn sàng để train mô hình Stage 2 Fusion.")
