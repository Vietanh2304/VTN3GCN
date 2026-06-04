"""
Evaluate fusion models 10x on different test feature extractions.
Reports Top-1 and Top-5 accuracy as mean ± std.

Full ablation: Stage 2 (12 configs) + Stage 3 (4 configs) = 16 fusion configs.
"""
import os, sys
sys.path.insert(0, '/media/ibmelab/ibme31/VTN3GCN_CBAM')
os.chdir('/media/ibmelab/ibme31/VTN3GCN_CBAM')

import json
import glob
import torch
import torch.nn.functional as F
import numpy as np
from modelling.fusion_models import FusionClassifier, LateFusionClassifier

DEVICE = 'cuda:1'
FEATURES_DIR = 'features/stage1_10x'
N_RUNS = 10
NUM_CLASSES = 199


def topk_accuracy(logits, labels, k=1):
    """Compute top-k accuracy."""
    _, top_indices = logits.topk(k, dim=-1)  # (B, k)
    correct = (top_indices == labels.unsqueeze(-1)).any(dim=-1).float()
    return correct.mean().item()


def load_fusion_model(result_dir):
    """Load fusion model from result_dir based on its args."""
    ckpt_path = os.path.join(result_dir, 'best_checkpoint.pth')
    if not os.path.exists(ckpt_path):
        return None, None

    ckpt = torch.load(ckpt_path, map_location='cpu')
    args = ckpt['args']

    method = args['method']
    views = args['views']
    num_views = len(views)
    dim = args.get('dim', 1024)
    dropout = args.get('dropout', 0.3)

    # Determine fusion type
    if method in ('late_avg', 'late_weighted'):
        model = LateFusionClassifier(
            method=method, num_classes=NUM_CLASSES, dim=dim,
            num_views=num_views, dropout=dropout
        )
    else:
        model = FusionClassifier(
            method=method, num_classes=NUM_CLASSES, dim=dim,
            num_views=num_views, dropout=dropout
        )

    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(DEVICE).eval()
    return model, views


@torch.no_grad()
def eval_one_run(model, features_dict, views, device):
    """Evaluate on one test feature dict, return (top1, top5)."""
    view_feats = [features_dict[v].to(device) for v in views]
    labels = features_dict['labels'].to(device)
    logits = model(view_feats)
    top1 = topk_accuracy(logits, labels, k=1)
    top5 = topk_accuracy(logits, labels, k=5)
    return top1, top5


def find_all_configs():
    """Find all Stage 2/3 configs that have best_checkpoint.pth."""
    configs = []
    for stage_dir in ['results/Stage2', 'results/Stage3']:
        stage = os.path.basename(stage_dir)
        for d in sorted(glob.glob(f'{stage_dir}/*/')):
            ckpt = os.path.join(d, 'best_checkpoint.pth')
            if os.path.exists(ckpt):
                name = os.path.basename(d.rstrip('/'))
                configs.append((stage, name, d))
    return configs


def main():
    print(f"{'='*70}")
    print(f"  10x Evaluation — Fusion Models (Top-1 + Top-5)")
    print(f"{'='*70}\n")

    # Verify 10x test files
    missing = []
    for i in range(1, N_RUNS + 1):
        p = f'{FEATURES_DIR}/test_run{i}.pt'
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        print(f"ERROR: Missing test files:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    # Load all 10 test runs
    print(f"Loading {N_RUNS} test feature files...")
    test_runs = []
    for i in range(1, N_RUNS + 1):
        d = torch.load(f'{FEATURES_DIR}/test_run{i}.pt', map_location='cpu')
        test_runs.append(d)
    print(f"  Loaded.\n")

    # Find all configs
    configs = find_all_configs()
    print(f"Found {len(configs)} fusion configs to evaluate.\n")

    # Eval
    summary = []
    for stage, name, result_dir in configs:
        print(f"{'─'*70}")
        print(f"  {stage} / {name}")
        print(f"{'─'*70}")

        model, views = load_fusion_model(result_dir)
        if model is None:
            print(f"  SKIP")
            continue

        top1s, top5s = [], []
        for i, td in enumerate(test_runs, 1):
            t1, t5 = eval_one_run(model, td, views, DEVICE)
            top1s.append(t1)
            top5s.append(t5)
            print(f"  Run {i:2d}: top1={t1:.4f}  top5={t5:.4f}")

        top1_t = torch.tensor(top1s)
        top5_t = torch.tensor(top5s)
        m1, s1 = top1_t.mean().item(), top1_t.std().item()
        m5, s5 = top5_t.mean().item(), top5_t.std().item()
        print(f"\n  Top-1: {m1*100:.2f} ± {s1*100:.4f}")
        print(f"  Top-5: {m5*100:.2f} ± {s5*100:.4f}")

        summary.append({
            'stage': stage,
            'config': name,
            'views': views,
            'method': name.split('_', 1)[1] if '_' in name else name,
            'top1_mean': m1, 'top1_std': s1,
            'top5_mean': m5, 'top5_std': s5,
            'top1_runs': top1s,
            'top5_runs': top5s,
        })

        del model
        torch.cuda.empty_cache()

    # Save JSON
    out_json = 'results/10x_eval_summary.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n\nSaved: {out_json}")

    # ============================================================
    # Print results table (paper-style)
    # ============================================================
    print(f"\n\n{'='*80}")
    print(f"  RESULTS TABLE — paper format")
    print(f"{'='*80}\n")

    # Stage 1 results (manual, single-eval — from training logs)
    stage1_results = [
        ("Center (single view)",  0.9077, None),  # MDramp30
        ("Left (single view)",    0.7636, None),
        ("Right (single view)",   0.8571, None),
    ]

    # Header
    print(f"{'Method':<35} | {'Top-1 Accuracy (%)':<22} | {'Top-5 Accuracy (%)':<22}")
    print(f"{'─'*35}-+-{'─'*22}-+-{'─'*22}")

    # Stage 1
    for name, top1, _ in stage1_results:
        top1_str = f"{top1*100:.2f}"
        print(f"{name:<35} | {top1_str:<22} | {'N/A':<22}")

    # Stage 2
    s2 = [s for s in summary if s['stage'] == 'Stage2']
    s2.sort(key=lambda x: x['config'])
    for s in s2:
        method_str = f"Stage2 {s['config']}"
        t1_str = f"{s['top1_mean']*100:.2f} ± {s['top1_std']*100:.4f}"
        t5_str = f"{s['top5_mean']*100:.2f} ± {s['top5_std']*100:.4f}"
        print(f"{method_str:<35} | {t1_str:<22} | {t5_str:<22}")

    # Stage 3
    s3 = [s for s in summary if s['stage'] == 'Stage3']
    s3.sort(key=lambda x: -x['top1_mean'])  # best first
    for s in s3:
        method_str = f"Stage3 {s['config']}"
        t1_str = f"{s['top1_mean']*100:.2f} ± {s['top1_std']*100:.4f}"
        t5_str = f"{s['top5_mean']*100:.2f} ± {s['top5_std']*100:.4f}"
        print(f"{method_str:<35} | {t1_str:<22} | {t5_str:<22}")

    # Save CSV
    csv_path = 'results/10x_eval_table.csv'
    with open(csv_path, 'w') as f:
        f.write("Method,Top-1 Mean (%),Top-1 Std (%),Top-5 Mean (%),Top-5 Std (%)\n")
        # Stage 1
        for name, top1, _ in stage1_results:
            f.write(f"{name},{top1*100:.2f},,,\n")
        # Stage 2
        for s in s2:
            f.write(f"Stage2 {s['config']},{s['top1_mean']*100:.2f},{s['top1_std']*100:.4f},"
                    f"{s['top5_mean']*100:.2f},{s['top5_std']*100:.4f}\n")
        # Stage 3
        for s in s3:
            f.write(f"Stage3 {s['config']},{s['top1_mean']*100:.2f},{s['top1_std']*100:.4f},"
                    f"{s['top5_mean']*100:.2f},{s['top5_std']*100:.4f}\n")
    print(f"\n  CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
