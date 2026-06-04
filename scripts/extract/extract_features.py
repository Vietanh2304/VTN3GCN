"""
Extract features từ 3 best_ckpt Stage 1 dùng strict=False load (bypass load_model patch).
Output: features_stage1/{train,val,test}.pt với dict {center, left, right, labels}
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

DEVICE = 'cuda:1'
BATCH_SIZE = 16
NUM_WORKERS = 6
OUTPUT_DIR = 'features/stage1'
CSV_3VIEW = '/mnt/sda1/VSLR_Storage/vsl/label1-200/label/labelThreeView'
TMP_CSV = '/tmp/extract_per_view_csv'
SPLITS = ['train', 'val', 'test']

VIEW_CONFIGS = {
    'center': {
        'cfg_path': 'configs/Stage1_SingleView/Stage1_Center_VSL199.yaml',
        'ckpt_path': 'checkpoints/vtn_att_poseflow/Stage1 Center CBAM_T1 VSL199/best_checkpoints.pth',
        'model_cls': VTNHCPF,
        'col': 'center',
    },
    'left': {
        'cfg_path': 'configs/Stage1_SingleView/Stage1_Left_VSL199.yaml',
        'ckpt_path': 'checkpoints/VTNGCN/Stage1 Left CBAM_T1 MDramp30 VSL199/best_checkpoints.pth',
        'model_cls': VTNHCPF_GCN,
        'col': 'left',
    },
    'right': {
        'cfg_path': 'configs/Stage1_SingleView/Stage1_Right_VSL199.yaml',
        'ckpt_path': 'checkpoints/VTNGCN/Stage1 Right CBAM_T1 MDramp30 VSL199/best_checkpoints.pth',
        'model_cls': VTNHCPF_GCN,
        'col': 'right',
    },
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Generate per-view CSV
print("=== Step 1: Per-view CSV from labelThreeView ===")
for split in SPLITS:
    df_3view = pd.read_csv(f'{CSV_3VIEW}/{split}_labels.csv')
    for view, conf in VIEW_CONFIGS.items():
        view_dir = f'{TMP_CSV}/{view}'
        os.makedirs(view_dir, exist_ok=True)
        df_3view[[conf['col'], 'label']].rename(columns={conf['col']: 'file_name'}).to_csv(
            f'{view_dir}/{split}_labels.csv', index=False)
    print(f"  {split}: {len(df_3view)} samples")

# Step 2: Extract per view
results = {split: {'center': None, 'left': None, 'right': None, 'labels': None} for split in SPLITS}

for view, conf in VIEW_CONFIGS.items():
    print(f"\n=== Step 2: Extract {view} ===")
    cfg = yaml.safe_load(open(conf['cfg_path']))
    cfg['training']['pretrained'] = False
    cfg['training']['device'] = DEVICE
    cfg['training']['batch_size'] = BATCH_SIZE
    cfg['training']['num_workers'] = NUM_WORKERS
    cfg['training']['prefetch_factor'] = 2
    
    # Build model + strict load
    model = conf['model_cls'](**cfg['model'], sequence_length=cfg['data']['num_output_frames'])
    state_dict = torch.load(conf['ckpt_path'], map_location='cpu')
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing: print(f"    Missing: {missing[:5]}")
    if unexpected: print(f"    Unexpected: {unexpected[:5]}")
    
    model = model.to(DEVICE).eval()
    for p in model.parameters(): p.requires_grad = False
    
    for split in SPLITS:
        sub_df = pd.read_csv(f'{TMP_CSV}/{view}/{split}_labels.csv')
        loader = build_dataloader(cfg, split, is_train=False, model=model, labels=sub_df)
        
        feats_list, labels_list = [], []
        for batch in tqdm(loader, desc=f'  {view} {split}'):
            features, labels = batch[0], batch[1]
            features = {k: v.to(DEVICE) for k, v in features.items() if hasattr(v, 'to')}
            
            with torch.no_grad():
                clip = features['clip']
                poseflow = features['poseflow']
                b, t, x, c, h, w = clip.size()
                z_rgb = model.feature_extractor(clip.view(b, t*x, c, h, w)).view(b, t, -1)
                
                if view == 'center':
                    feat_seq = model.forward_features(features=z_rgb, poseflow=poseflow)
                else:
                    keypoints = features['keypoints']
                    z_kp = model.feature_extractor_gcn(keypoints)
                    feat_seq = model.forward_features(
                        features=z_rgb, poseflow=poseflow, features_keypoint=z_kp)
                
                feat = feat_seq.mean(1).cpu()
            
            feats_list.append(feat)
            labels_list.append(labels)
        
        feats_all = torch.cat(feats_list, dim=0)
        labels_all = torch.cat(labels_list, dim=0)
        results[split][view] = feats_all
        
        if results[split]['labels'] is None:
            results[split]['labels'] = labels_all
        else:
            assert torch.equal(results[split]['labels'], labels_all), \
                f"Labels mismatch {view} {split}!"
        
        print(f"    -> features {feats_all.shape}, labels {labels_all.shape}")
    
    del model
    torch.cuda.empty_cache()

# Step 3: Save
print("\n=== Step 3: Save ===")
for split in SPLITS:
    out_path = f'{OUTPUT_DIR}/{split}.pt'
    torch.save(results[split], out_path)
    sz = os.path.getsize(out_path) / 1024 / 1024
    print(f"  {out_path}: center {results[split]['center'].shape}, "
          f"left {results[split]['left'].shape}, right {results[split]['right'].shape} ({sz:.1f} MB)")

print("\nDone! Features ready for Stage 2 fusion training.")
