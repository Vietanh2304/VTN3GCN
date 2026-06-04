"""
Generate Figure 6 for paper: qualitative error analysis.
Layout per case (1 row, 10 cells):
  [Ref GT (full video)] [Ref Pred (full video)] 
  [Hand A: t=4, 8, 11, 14] [Hand B: t=4, 8, 11, 14]
"""
import torch
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils.misc import load_config
from dataset.dataloader import build_dataloader
from modelling.vtn_att_poseflow_model import VTNHCPF
from run_qualitative_analysis import AttentionPoolViz

CONFIG_PATH     = "configs/Stage1_SingleView/Stage1_Center_VSL199.yaml"
CHECKPOINT_PATH = "checkpoints/vtn_att_poseflow/Stage1 Center CBAM_T1 VSL199/best_FINAL_test90.77_lr1e-4.pth"
TEST_CSV        = "/mnt/sda1/VSLR_Storage/vsl/label1-200/label/labelCenter/test_labels.csv"
TRAIN_CSV       = "/mnt/sda1/VSLR_Storage/vsl/label1-200/label/labelCenter/train_labels.csv"
VIDEO_DIR       = "/mnt/sda1/VSLR_Storage/vsl/videos"
FAILURES_CSV    = "./qual_results/failures.csv"
OUTPUT_PATH     = "./qual_results/figure6_qualitative.png"
DEVICE          = "cuda:1"
FRAME_INDICES   = [4, 14]

CASES = [
    {'gt': 114, 'pred': 113, 'label': '(a)'},
    {'gt': 41,  'pred': 150, 'label': '(b)'},
    {'gt': 134, 'pred': 146, 'label': '(c)'},
]


def get_middle_frame(video_path):
    if not os.path.exists(video_path):
        print(f"  ! Khong tim thay: {video_path}")
        return None
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def get_reference_frame_from_train(label_id, train_df, video_dir):
    """Lay 1 frame giua tu video train co label_id (cho Ref Pred)."""
    match = train_df[train_df['label_id'] == label_id]
    center = match[match['file_name'].str.contains('_center_', na=False)]
    if len(center) == 0:
        center = match
    if len(center) == 0:
        return None
    video_name = center.iloc[0]['file_name']
    return get_middle_frame(os.path.join(video_dir, video_name))


def get_test_video_path(sample_idx, test_df, video_dir):
    """Map sample_idx -> video file path."""
    if sample_idx >= len(test_df):
        return None
    filename = test_df.iloc[sample_idx]['file_name']
    return os.path.join(video_dir, filename)


def overlay_attention(frame_uint8, heatmap):
    h, w = frame_uint8.shape[:2]
    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax > hmin:
        heatmap = (heatmap - hmin) / (hmax - hmin)
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_color = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return (0.40 * heatmap_color + 0.60 * frame_uint8).astype(np.uint8)


def main():
    if not os.path.exists(FAILURES_CSV):
        print(f"ERROR: {FAILURES_CSV} khong ton tai.")
        return

    cfg = load_config(CONFIG_PATH)
    cfg['training']['test'] = True
    cfg['training']['pretrained'] = False
    cfg['data']['base_url'] = '/mnt/sda1/VSLR_Storage/vsl'

    device = torch.device(DEVICE)
    model = VTNHCPF(**cfg['model'], sequence_length=cfg['data']['num_output_frames'])
    state_dict = torch.load(CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print("Model loaded")

    test_loader = build_dataloader(cfg, 'test', is_train=False, model=model)
    test_df = pd.read_csv(TEST_CSV)
    train_df = pd.read_csv(TRAIN_CSV)
    failures_df = pd.read_csv(FAILURES_CSV)

    for case in CASES:
        match = failures_df[
            (failures_df['gt_class'] == case['gt']) &
            (failures_df['pred_class'] == case['pred'])
        ]
        if len(match) == 0:
            print(f"! Khong tim thay failure cho GT={case['gt']} Pred={case['pred']}")
            return
        best = match.sort_values('confidence', ascending=False).iloc[0]
        case['sample_idx'] = int(best['sample_idx'])
        case['confidence'] = float(best['confidence'])
        print(f"Case {case['label']}: GT=gloss_{case['gt']} Pred=gloss_{case['pred']} "
              f"sample_idx={case['sample_idx']} conf={case['confidence']:.3f}")

    viz = AttentionPoolViz(model)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    case_data = []
    for case in CASES:
        print(f"\nProcessing {case['label']}...")

        # Ref GT: lay tu chinh video failure (full video)
        test_video_path = get_test_video_path(case['sample_idx'], test_df, VIDEO_DIR)
        ref_gt = get_middle_frame(test_video_path) if test_video_path else None

        # Ref Pred: lay tu train (gloss khac)
        ref_pred = get_reference_frame_from_train(case['pred'], train_df, VIDEO_DIR)

        if ref_gt is None or ref_pred is None:
            print(f"  ! Bo qua case {case['label']} (thieu reference)")
            continue

        sample = test_loader.dataset[case['sample_idx']]
        clip, poseflow, _ = sample
        batch = {
            'clip': clip.unsqueeze(0).to(device),
            'poseflow': poseflow.unsqueeze(0).to(device)
        }
        attn_map = viz(batch).cpu()

        case_data.append({
            'label': case['label'],
            'gt': case['gt'],
            'pred': case['pred'],
            'confidence': case['confidence'],
            'ref_gt': ref_gt,
            'ref_pred': ref_pred,
            'clip': clip,
            'attn_map': attn_map,
        })
        print("  OK")

    viz.remove_hooks()

    if len(case_data) == 0:
        print("Khong process duoc case nao.")
        return

    # Plot: each case = 1 row
    # Cols: [label] [ref_gt] [ref_pred] [Hand A x 4] [Hand B x 4]
    n_cases = len(case_data)
    n_frames = len(FRAME_INDICES)
    total_cols = 1 + 2 + 2 * n_frames  # = 11

    fig = plt.figure(figsize=(total_cols * 1.5, n_cases * 1.9))
    gs = gridspec.GridSpec(
        n_cases, total_cols, figure=fig,
        hspace=0.30, wspace=0.10,
        width_ratios=[0.25] + [1.3, 1.3] + [1.0] * (2 * n_frames)
    )

    for ci, cd in enumerate(case_data):
        # Col 0: case label
        ax = fig.add_subplot(gs[ci, 0])
        ax.text(0.5, 0.5, cd['label'],
                fontsize=18, fontweight='bold',
                ha='center', va='center')
        ax.axis('off')

        # Col 1: Ref GT (from test video, full frame)
        ax = fig.add_subplot(gs[ci, 1])
        ax.imshow(cd['ref_gt'])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color('darkgreen'); s.set_linewidth(2)
        if ci == 0:
            ax.set_title("Ref. GT", fontsize=10, fontweight='bold', color='darkgreen')
        ax.set_xlabel(f"gloss_{cd['gt']}", fontsize=9, color='darkgreen')

        # Col 2: Ref Pred (from train, full frame)
        ax = fig.add_subplot(gs[ci, 2])
        ax.imshow(cd['ref_pred'])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color('darkred'); s.set_linewidth(2)
        if ci == 0:
            ax.set_title("Ref. Pred", fontsize=10, fontweight='bold', color='darkred')
        ax.set_xlabel(f"gloss_{cd['pred']}", fontsize=9, color='darkred')

        # Attention overlays cho 2 hands
        clip = cd['clip']
        attn_map = cd['attn_map']

        for crop_idx in range(2):
            crop_clip = clip[:, crop_idx, :, :, :]
            col_offset = 3 + crop_idx * n_frames

            for fi, t_idx in enumerate(FRAME_INDICES):
                col = col_offset + fi
                frame = crop_clip[t_idx].permute(1, 2, 0).cpu().numpy()
                frame = np.clip(frame * std + mean, 0, 1)
                frame_uint8 = (frame * 255).astype(np.uint8)
                heatmap = attn_map[t_idx * 2 + crop_idx].numpy()
                overlay = overlay_attention(frame_uint8, heatmap)

                ax = fig.add_subplot(gs[ci, col])
                ax.imshow(overlay)
                ax.set_xticks([]); ax.set_yticks([])
                if ci == 0:
                    if fi == 0:
                        hand_name = 'Hand A' if crop_idx == 0 else 'Hand B'
                        ax.set_title(f"{hand_name}\nt={t_idx}", fontsize=9)
                    else:
                        ax.set_title(f"t={t_idx}", fontsize=9)

    plt.suptitle(
        "Figure 6 — Qualitative error analysis on high-confidence failures.\n"
        "Ref. GT: middle frame of the failure video (test). "
        "Ref. Pred: a reference video for the gloss model wrongly predicted (train). "
        "Hand A / Hand B: AttentionPool2D weights on the two hand crops.",
        fontsize=9, y=1.02
    )

    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
