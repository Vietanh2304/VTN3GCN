"""
train_fusion.py — Train fusion classifier on pre-extracted Stage 1 features.

Supports both feature-level fusion (concat, addweighted, crossattn, gmu)
and late fusion (late_avg, late_weighted).
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modelling.fusion_models import build_classifier, is_late_fusion, ALL_METHODS


def load_split(features_dir, split, views):
    d = torch.load(os.path.join(features_dir, f"{split}.pt"), map_location="cpu")
    view_tensors = [d[v].float() for v in views]
    labels = d["labels"].long()
    n = labels.shape[0]
    for v, t in zip(views, view_tensors):
        assert t.shape[0] == n, f"{split}/{v}: expected {n} samples, got {t.shape[0]}"
    return view_tensors, labels


class MultiViewFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, view_tensors, labels):
        self.views = view_tensors
        self.labels = labels
        self.num_views = len(view_tensors)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return [v[idx] for v in self.views], self.labels[idx]


def collate(batch):
    views_list = list(zip(*[item[0] for item in batch]))
    views_stacked = [torch.stack(v, dim=0) for v in views_list]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return views_stacked, labels


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for views, labels in loader:
        views = [v.to(device, non_blocking=True) for v in views]
        labels = labels.to(device, non_blocking=True)
        logits = model(views)
        loss_sum += ce(logits, labels).item()
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.numel()
    return correct / total, loss_sum / total


def train_one_epoch(model, loader, optimizer, device, label_smoothing=0.1, grad_clip=1.0):
    model.train()
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    loss_sum = correct = total = 0
    for views, labels in loader:
        views = [v.to(device, non_blocking=True) for v in views]
        labels = labels.to(device, non_blocking=True)
        logits = model(views)
        loss = ce(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
    return correct / total, loss_sum / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str, default="features/stage1")
    parser.add_argument("--views", nargs="+", required=True,
                        choices=["front", "left", "right"])
    parser.add_argument("--method", type=str, required=True, choices=ALL_METHODS)
    parser.add_argument("--num_classes", type=int, default=199)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    fusion_type = "late" if is_late_fusion(args.method) else "feature"

    print(f"\n{'='*70}")
    print(f"Fusion train | views={args.views} | method={args.method} ({fusion_type})")
    print(f"output_dir={args.output_dir} | device={device}")
    print(f"{'='*70}\n")

    print("Loading features...")
    tr_views, tr_labels = load_split(args.features_dir, "train", args.views)
    va_views, va_labels = load_split(args.features_dir, "val",   args.views)
    te_views, te_labels = load_split(args.features_dir, "test",  args.views)
    print(f"  train: {tr_labels.shape[0]} | val: {va_labels.shape[0]} | test: {te_labels.shape[0]}")

    tr_ds = MultiViewFeatureDataset(tr_views, tr_labels)
    va_ds = MultiViewFeatureDataset(va_views, va_labels)
    te_ds = MultiViewFeatureDataset(te_views, te_labels)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, collate_fn=collate,
                           pin_memory=True, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, collate_fn=collate, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, collate_fn=collate, pin_memory=True)

    # Build (auto-detect feature vs late fusion)
    model = build_classifier(
        method=args.method,
        num_classes=args.num_classes,
        dim=args.dim,
        num_views=len(args.views),
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    fusion_out_dim = model.out_dim if is_late_fusion(args.method) else model.fusion.out_dim
    print(f"  type={fusion_type} | params: {n_params/1e6:.2f}M | out_dim={fusion_out_dim}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(1, args.warmup_epochs)
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_acc = 0.0
    best_epoch = -1
    epochs_since_best = 0
    history = []
    best_ckpt_path = os.path.join(args.output_dir, "best_checkpoint.pth")

    t_start = time.time()
    for epoch in range(args.epochs):
        ep_t = time.time()
        tr_acc, tr_loss = train_one_epoch(
            model, tr_loader, optimizer, device,
            label_smoothing=args.label_smoothing, grad_clip=args.grad_clip,
        )
        va_acc, va_loss = evaluate(model, va_loader, device)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch, "train_acc": tr_acc, "train_loss": tr_loss,
            "val_acc": va_acc, "val_loss": va_loss, "lr": lr_now,
        })

        is_best = va_acc > best_val_acc
        if is_best:
            best_val_acc = va_acc
            best_epoch = epoch
            epochs_since_best = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_acc": va_acc, "args": vars(args)}, best_ckpt_path)
        else:
            epochs_since_best += 1

        print(f"E{epoch:03d} | tr_acc={tr_acc:.4f} tr_loss={tr_loss:.4f} | "
              f"va_acc={va_acc:.4f} va_loss={va_loss:.4f} | "
              f"lr={lr_now:.2e} | best={best_val_acc:.4f}@{best_epoch} | "
              f"{time.time()-ep_t:.1f}s {'*' if is_best else ''}")

        if epochs_since_best >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={args.patience}).")
            break

    total_time = time.time() - t_start

    print(f"\nLoading best ckpt from epoch {best_epoch} (val_acc={best_val_acc:.4f})...")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_acc, test_loss = evaluate(model, te_loader, device)
    print(f"TEST: acc={test_acc:.4f} loss={test_loss:.4f}")

    result = {
        "views": args.views,
        "method": args.method,
        "fusion_type": fusion_type,
        "num_views": len(args.views),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "fusion_out_dim": fusion_out_dim,
        "num_params": n_params,
        "total_time_sec": total_time,
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone in {total_time/60:.1f}min. Saved to {args.output_dir}")
    print(f"  best_val_acc = {best_val_acc:.4f} @ epoch {best_epoch}")
    print(f"  test_acc     = {test_acc:.4f}")


if __name__ == "__main__":
    main()
