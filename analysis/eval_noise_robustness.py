"""
eval_noise_robustness.py  —  R3-C3: Pose Noise Robustness Study

Inject Gaussian noise (scaled to feature std) vào GCN feature embeddings,
evaluate LCR_concat fusion model (Stage3) trên 10 test runs.

Noise model: noisy_feat = feat + N(0, sigma * std(feat))
  - sigma=0.0  → clean baseline
  - sigma=0.01 → very mild (sub-pixel pose jitter)
  - sigma=0.05 → moderate (realistic mediapipe noise)
  - sigma=0.10 → strong
  - sigma=0.20 → severe (occlusion-level degradation)

Usage:
    python eval_noise_robustness.py
    python eval_noise_robustness.py --device cuda:0 --stage Stage2
"""
import os, sys, argparse, json
sys.path.insert(0, '/media/ibmelab/ibme31/VTN3GCN_CBAM')
os.chdir('/media/ibmelab/ibme31/VTN3GCN_CBAM')

import torch
import numpy as np
from modelling.fusion_models import FusionClassifier, LateFusionClassifier

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_DIR  = 'features/stage1_10x'
N_RUNS        = 10
NUM_CLASSES   = 199
TARGET_CONFIG = 'LCR_concat'   # best Stage3 model
NOISE_LEVELS  = [0.0, 0.01, 0.05, 0.10, 0.20]
NOISY_VIEWS   = ['left', 'right']   # GCN views; center = VTN (no GCN keypoints)
# ──────────────────────────────────────────────────────────────────────────────


def topk_accuracy(logits, labels, k=1):
    _, top_idx = logits.topk(k, dim=-1)
    return (top_idx == labels.unsqueeze(-1)).any(dim=-1).float().mean().item()


def load_fusion_model(stage, config_name, device):
    ckpt_path = f'results/{stage}/{config_name}/best_checkpoint.pth'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt   = torch.load(ckpt_path, map_location='cpu')
    args   = ckpt['args']
    method = args['method']
    views  = args['views']
    dim    = args.get('dim', 1024)
    dropout = args.get('dropout', 0.3)

    if method in ('late_avg', 'late_weighted'):
        model = LateFusionClassifier(
            method=method, num_classes=NUM_CLASSES,
            dim=dim, num_views=len(views), dropout=dropout)
    else:
        model = FusionClassifier(
            method=method, num_classes=NUM_CLASSES,
            dim=dim, num_views=len(views), dropout=dropout)

    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    return model, views


def inject_noise(features_dict, views_to_perturb, sigma, device, seed=None):
    """Return new dict with Gaussian noise added to specified views.
    Noise scale = sigma * per-feature std (relative perturbation).
    """
    if seed is not None:
        torch.manual_seed(seed)

    noisy = {}
    for k, v in features_dict.items():
        if k in views_to_perturb and sigma > 0.0:
            v_dev = v.to(device).float()
            # Per-feature std across samples → shape (1, D)
            feat_std = v_dev.std(dim=0, keepdim=True).clamp(min=1e-6)
            noise    = torch.randn_like(v_dev) * sigma * feat_std
            noisy[k] = (v_dev + noise).cpu()
        else:
            noisy[k] = v
    return noisy


@torch.no_grad()
def eval_one_run(model, features_dict, views, device):
    view_feats = [features_dict[v].to(device) for v in views]
    labels     = features_dict['labels'].to(device)
    logits     = model(view_feats)
    return (topk_accuracy(logits, labels, k=1),
            topk_accuracy(logits, labels, k=5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--stage',  default='Stage3')
    parser.add_argument('--config', default=TARGET_CONFIG)
    args = parser.parse_args()

    device = args.device
    stage  = args.stage
    config = args.config

    print(f"\n{'='*65}")
    print(f"  R3-C3 Noise Robustness — {stage}/{config}")
    print(f"  Device : {device}")
    print(f"  Perturbed views : {NOISY_VIEWS}  (center kept clean)")
    print(f"  Noise levels σ  : {NOISE_LEVELS}")
    print(f"{'='*65}\n")

    # Load model
    model, views = load_fusion_model(stage, config, device)
    print(f"Loaded model.  Views used: {views}\n")

    # Verify & load 10 test runs
    test_runs = []
    for i in range(1, N_RUNS + 1):
        p = f'{FEATURES_DIR}/test_run{i}.pt'
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")
        test_runs.append(torch.load(p, map_location='cpu'))
    print(f"Loaded {N_RUNS} test feature files.\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    results = {}   # sigma → {top1: [10 floats], top5: [10 floats]}

    for sigma in NOISE_LEVELS:
        top1s, top5s = [], []
        for run_idx, td in enumerate(test_runs):
            # Fixed seed per (sigma, run) for reproducibility
            noisy_td = inject_noise(td, NOISY_VIEWS, sigma, device,
                                    seed=42 + run_idx)
            t1, t5 = eval_one_run(model, noisy_td, views, device)
            top1s.append(t1)
            top5s.append(t5)

        m1 = np.mean(top1s);  s1 = np.std(top1s)
        m5 = np.mean(top5s);  s5 = np.std(top5s)
        results[sigma] = dict(top1_runs=top1s, top5_runs=top5s,
                              top1_mean=m1, top1_std=s1,
                              top5_mean=m5, top5_std=s5)

        tag = "← baseline" if sigma == 0.0 else ""
        print(f"  σ={sigma:.2f}  |  Top-1: {m1*100:.2f} ± {s1*100:.3f}%"
              f"  |  Top-5: {m5*100:.2f} ± {s5*100:.3f}%  {tag}")

    # ── Pretty table ──────────────────────────────────────────────────────────
    baseline_top1 = results[0.0]['top1_mean']
    print(f"\n{'─'*65}")
    print(f"  {'Noise σ':<12} {'Top-1 (%)':<22} {'Top-5 (%)':<22} {'Δ Top-1'}")
    print(f"{'─'*65}")
    for sigma in NOISE_LEVELS:
        r   = results[sigma]
        m1  = r['top1_mean'];  s1 = r['top1_std']
        m5  = r['top5_mean'];  s5 = r['top5_std']
        delta = (m1 - baseline_top1) * 100
        delta_str = f"{delta:+.2f}%" if sigma > 0 else "—"
        print(f"  {sigma:<12.2f} {m1*100:.2f} ± {s1*100:.3f}%       "
              f"{m5*100:.2f} ± {s5*100:.3f}%       {delta_str}")
    print(f"{'─'*65}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs('results', exist_ok=True)
    out_json = f'results/noise_robustness_{stage}_{config}.json'
    out_csv  = f'results/noise_robustness_{stage}_{config}.csv'

    # JSON
    save_data = {
        'stage': stage, 'config': config,
        'noisy_views': NOISY_VIEWS,
        'noise_levels': NOISE_LEVELS,
        'results': {str(s): v for s, v in results.items()}
    }
    with open(out_json, 'w') as f:
        json.dump(save_data, f, indent=2)

    # CSV (paper-ready)
    with open(out_csv, 'w') as f:
        f.write("noise_sigma,top1_mean,top1_std,top5_mean,top5_std,delta_top1\n")
        for sigma in NOISE_LEVELS:
            r = results[sigma]
            delta = (r['top1_mean'] - baseline_top1) * 100
            f.write(f"{sigma},"
                    f"{r['top1_mean']*100:.4f},{r['top1_std']*100:.4f},"
                    f"{r['top5_mean']*100:.4f},{r['top5_std']*100:.4f},"
                    f"{delta:.4f}\n")

    print(f"\nSaved:\n  {out_json}\n  {out_csv}")
    print("\nDone.")


if __name__ == '__main__':
    main()
