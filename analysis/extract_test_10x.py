"""
Extract test features 10 times with different random seeds.
Used for 10x evaluation of fusion methods (matches paper protocol).

Each run extracts test features with different temporal sampling (random),
saving to features/stage1_10x/test_run{i}.pt
"""
import os, sys
sys.path.insert(0, '/media/ibmelab/ibme31/VTN3GCN_CBAM')
os.chdir('/media/ibmelab/ibme31/VTN3GCN_CBAM')

import torch
import numpy as np
import random
import pandas as pd
import yaml
from tqdm import tqdm
from modelling.vtn_att_poseflow_model import VTNHCPF, VTNHCPF_GCN
from dataset.dataloader import build_dataloader

DEVICE = 'cuda:1'
BATCH_SIZE = 16
NUM_WORKERS = 6
OUTPUT_DIR = 'features/stage1_10x'
CSV_3VIEW = '/mnt/sda1/VSLR_Storage/vsl/label1-200/label/labelThreeView'
TMP_CSV = '/tmp/extract_per_view_csv'
N_RUNS = 10

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

# Reuse per-view CSV from main extract script
print("=== Loading per-view CSVs ===")
for view in VIEW_CONFIGS.keys():
    csv_path = f'{TMP_CSV}/{view}/test_labels.csv'
    if not os.path.exists(csv_path):
        # Re-generate if missing
        df_3view = pd.read_csv(f'{CSV_3VIEW}/test_labels.csv')
        view_dir = f'{TMP_CSV}/{view}'
        os.makedirs(view_dir, exist_ok=True)
        df_3view[[VIEW_CONFIGS[view]['col'], 'label']].rename(
            columns={VIEW_CONFIGS[view]['col']: 'file_name'}).to_csv(
            csv_path, index=False)
    print(f"  {view}: {csv_path}")


# Load models ONCE
print("\n=== Loading 3 models ===")
models = {}
cfgs = {}
for view, conf in VIEW_CONFIGS.items():
    cfg = yaml.safe_load(open(conf['cfg_path']))
    cfg['training']['pretrained'] = False
    cfg['training']['device'] = DEVICE
    cfg['training']['batch_size'] = BATCH_SIZE
    cfg['training']['num_workers'] = NUM_WORKERS
    cfg['training']['prefetch_factor'] = 2
    cfgs[view] = cfg

    model = conf['model_cls'](**cfg['model'], sequence_length=cfg['data']['num_output_frames'])
    state_dict = torch.load(conf['ckpt_path'], map_location='cpu')
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  {view}: missing={len(missing)}, unexpected={len(unexpected)}")

    model = model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False
    models[view] = model


# Run extract 10 times
for run_i in range(N_RUNS):
    print(f"\n{'='*60}")
    print(f"  RUN {run_i + 1}/{N_RUNS}")
    print(f"{'='*60}")

    # Set different seed for each run -> different temporal sampling
    seed = 42 + run_i * 100
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    result = {'center': None, 'left': None, 'right': None, 'labels': None}

    for view, conf in VIEW_CONFIGS.items():
        sub_df = pd.read_csv(f'{TMP_CSV}/{view}/test_labels.csv')
        loader = build_dataloader(cfgs[view], 'test', is_train=False,
                                   model=models[view], labels=sub_df)

        feats_list, labels_list = [], []
        model = models[view]
        for batch in tqdm(loader, desc=f'  {view}'):
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
        result[view] = feats_all

        if result['labels'] is None:
            result['labels'] = labels_all
        else:
            assert torch.equal(result['labels'], labels_all), \
                f"Labels mismatch {view}!"

    out_path = f'{OUTPUT_DIR}/test_run{run_i + 1}.pt'
    torch.save(result, out_path)
    sz = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Saved {out_path} ({sz:.1f} MB)")

print(f"\n{'='*60}")
print(f"  DONE — 10 test runs saved to {OUTPUT_DIR}/")
print(f"{'='*60}")
