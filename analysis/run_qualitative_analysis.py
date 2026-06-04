"""
Bước 1 của pipeline qualitative analysis:
- Extract failure cases tu test set
- Visualize attention weights cua AttentionPool2D cho 10 high-conf failures
- Save: failures.csv + 10 anh failure_XX_*.png
"""
from analyze_failures import extract_failure_cases, analyze_confusion_pairs

import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils.misc import load_config
from dataset.dataloader import build_dataloader
from modelling.vtn_att_poseflow_model import VTNHCPF

CONFIG_PATH     = "configs/Stage1_SingleView/Stage1_Center_VSL199.yaml"
CHECKPOINT_PATH = "checkpoints/vtn_att_poseflow/Stage1 Center CBAM_T1 VSL199/best_FINAL_test90.77_lr1e-4.pth"
OUTPUT_DIR      = "./qual_results"
DEVICE          = "cuda:1"
FRAME_INDICES   = [4, 8, 11, 14]


class AttentionPoolViz:
    """Capture attention weights tu AttentionPool2D bang monkey-patch forward."""
    def __init__(self, model):
        self.model = model
        self.attention_weights = None
        self._spatial_size = None
        self.attn_pool = model.feature_extractor.cbam_pool.attn_pool
        self.original_forward = self.attn_pool.forward
        self._patch_forward()

    def _patch_forward(self):
        attn_pool = self.attn_pool
        storage = self

        def patched_forward(x):
            bt, c, h, w = x.shape
            tokens = x.flatten(2).transpose(1, 2)
            q = attn_pool.query.expand(bt, -1, -1)
            k = attn_pool.k_proj(tokens)
            v = attn_pool.v_proj(tokens)

            def reshape_heads(t):
                return t.view(bt, -1, attn_pool.num_heads, attn_pool.head_dim).transpose(1, 2)

            q = reshape_heads(q)
            k = reshape_heads(k)
            v = reshape_heads(v)

            attn = (q @ k.transpose(-2, -1)) * attn_pool.scale
            attn = attn.softmax(dim=-1)
            storage.attention_weights = attn.detach().clone()
            storage._spatial_size = (h, w)
            out = (attn @ v)
            out = out.transpose(1, 2).reshape(bt, 1, c)
            out = attn_pool.out_proj(out).squeeze(1)
            return out

        self.attn_pool.forward = patched_forward

    def __call__(self, batch):
        with torch.no_grad():
            _ = self.model(**batch)
        attn = self.attention_weights.mean(dim=1).squeeze(1)
        h, w = self._spatial_size
        return attn.view(-1, h, w)

    def remove_hooks(self):
        self.attn_pool.forward = self.original_forward


def plot_attention_overlay(batch, attn_map, gt_name, pred_name, save_path,
                            frame_indices=None):
    if frame_indices is None:
        frame_indices = FRAME_INDICES
    T = batch['clip'].shape[1]
    frame_indices = [i for i in frame_indices if i < T]
    n_cols = len(frame_indices)

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(4, n_cols, figsize=(n_cols * 2.5, 9))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    title_str = "GT: " + str(gt_name) + "  -->  Pred: " + str(pred_name)
    fig.suptitle(title_str, fontsize=13, fontweight='bold', color='red')

    crop_labels = ['Hand A (crop 0)', 'Hand B (crop 1)']
    newline = chr(10)

    for crop_idx in range(2):
        clip = batch['clip'][0, :, crop_idx, :, :, :]

        for fi, t_idx in enumerate(frame_indices):
            frame = clip[t_idx].permute(1, 2, 0).cpu().numpy()
            frame = np.clip(frame * std + mean, 0, 1)
            frame_uint8 = (frame * 255).astype(np.uint8)

            row_orig = crop_idx * 2
            row_attn = crop_idx * 2 + 1

            axes[row_orig, fi].imshow(frame_uint8)
            if fi == 0:
                ttl = crop_labels[crop_idx] + newline + "t=" + str(t_idx)
            else:
                ttl = "t=" + str(t_idx)
            axes[row_orig, fi].set_title(ttl, fontsize=8)
            axes[row_orig, fi].axis('off')

            h, w = frame_uint8.shape[:2]
            heatmap = attn_map[t_idx * 2 + crop_idx].cpu().numpy()
            hmin, hmax = heatmap.min(), heatmap.max()
            if hmax > hmin:
                heatmap = (heatmap - hmin) / (hmax - hmin)
            heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
            heatmap_color = cv2.applyColorMap(
                (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            overlay = (0.40 * heatmap_color + 0.60 * frame_uint8).astype(np.uint8)

            axes[row_attn, fi].imshow(overlay)
            attn_title = 'CBAM Attention' if fi == 0 else ''
            axes[row_attn, fi].set_title(attn_title, fontsize=8)
            axes[row_attn, fi].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: " + save_path)


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load config + model
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
    print("Model loaded: test_acc=90.77%")

    # Dataloader
    test_loader = build_dataloader(cfg, 'test', is_train=False, model=model)
    print(f"Test set: {len(test_loader.dataset)} samples")

    num_classes = cfg['model']['num_classes']
    class_names = [f"gloss_{i}" for i in range(num_classes)]

    # STEP 1: Extract failures
    print("\n=== STEP 1: Extracting failure cases ===")
    failures = extract_failure_cases(
        model=model,
        dataloader=test_loader,
        class_names=class_names,
        device=device,
        top_k=20
    )
    analyze_confusion_pairs(failures, top_n=10)
    pd.DataFrame(failures).to_csv(os.path.join(OUTPUT_DIR, "failures.csv"), index=False)
    print("Saved failures.csv")

    # STEP 2: Attention visualization
    print("\n=== STEP 2: Generating CBAM Attention Visualizations ===")
    viz = AttentionPoolViz(model)

    for i, failure in enumerate(failures[:10]):
        print(f"[{i+1}/10] GT: {failure['gt_name']} -> Pred: {failure['pred_name']}")
        sample = test_loader.dataset[failure['sample_idx']]
        clip, poseflow, label = sample
        batch = {
            'clip': clip.unsqueeze(0).to(device),
            'poseflow': poseflow.unsqueeze(0).to(device)
        }
        try:
            attn_map = viz(batch)
            print(f"    attn shape: {attn_map.shape}")
            plot_attention_overlay(
                batch=batch, attn_map=attn_map,
                gt_name=failure['gt_name'], pred_name=failure['pred_name'],
                save_path=os.path.join(
                    OUTPUT_DIR,
                    f"failure_{i+1:02d}_GT{failure['gt_name']}_PRED{failure['pred_name']}.png"
                )
            )
        except Exception as e:
            import traceback
            print(f"  Loi: {e}")
            traceback.print_exc()

    viz.remove_hooks()
    print(f"\n=== DONE! Results in {OUTPUT_DIR}/ ===")
