"""
Fusion modules for Stage 2 (pairwise) and Stage 3 (triple) of VTN3GCN_CBAM.

Two families:
  A) Feature-level fusion (early/mid): fuse features -> 1 classifier head
     - ConcatFusion, AddWeightedFusion, CrossAttentionFusion, GMUFusion
     - wrapped in FusionClassifier

  B) Logit-level fusion (late): N classifiers (one per view) -> fuse logits
     - LateAvgFusion, LateWeightedFusion
     - wrapped in LateFusionClassifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# A) FEATURE-LEVEL FUSION MODULES
# ===========================================================================

class ConcatFusion(nn.Module):
    """Baseline: concatenate all view features along feature dim."""

    def __init__(self, dim: int = 1024, num_views: int = 2, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_views = num_views
        self.out_dim = dim * num_views
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, views):
        x = torch.cat(views, dim=-1)
        return self.dropout(x)


class AddWeightedFusion(nn.Module):
    """Softmax-weighted sum with learnable scalar per view."""

    def __init__(self, dim: int = 1024, num_views: int = 2, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_views = num_views
        self.out_dim = dim
        self.alpha = nn.Parameter(torch.zeros(num_views))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, views):
        w = F.softmax(self.alpha, dim=0)
        stacked = torch.stack(views, dim=0)
        out = (w.view(-1, 1, 1) * stacked).sum(dim=0)
        return self.dropout(out)


class CrossAttentionFusion(nn.Module):
    """Each view = 1 token, multi-head self-attention, then flatten tokens."""

    def __init__(self, dim: int = 1024, num_views: int = 2,
                 num_heads: int = 8, dropout: float = 0.1, ffn_ratio: int = 2):
        super().__init__()
        self.dim = dim
        self.num_views = num_views
        self.out_dim = dim * num_views

        self.view_pos = nn.Parameter(torch.zeros(1, num_views, dim))
        nn.init.trunc_normal_(self.view_pos, std=0.02)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, views):
        x = torch.stack(views, dim=1)
        x = x + self.view_pos
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x.flatten(1)


class GMUFusion(nn.Module):
    """Gated Multimodal Unit: per-view transform + content-dependent gate."""

    def __init__(self, dim: int = 1024, num_views: int = 2, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_views = num_views
        self.out_dim = dim
        self.h_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_views)])
        self.gate = nn.Linear(dim * num_views, num_views)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, views):
        h = [torch.tanh(self.h_proj[i](views[i])) for i in range(self.num_views)]
        h = torch.stack(h, dim=1)
        concat = torch.cat(views, dim=-1)
        z = F.softmax(self.gate(concat), dim=-1)
        out = (z.unsqueeze(-1) * h).sum(dim=1)
        return self.dropout(out)


# ===========================================================================
# B) LATE FUSION (logit-level)
# ===========================================================================

class LateAvgFusion(nn.Module):
    """Average logits across views. No fusion parameters."""

    def __init__(self, num_views: int = 2):
        super().__init__()
        self.num_views = num_views

    def forward(self, logits_list):
        # logits_list: list of [B, num_classes]
        stacked = torch.stack(logits_list, dim=0)        # [num_views, B, C]
        return stacked.mean(dim=0)                       # [B, C]


class LateWeightedFusion(nn.Module):
    """Softmax-weighted average of logits with learnable per-view scalars."""

    def __init__(self, num_views: int = 2):
        super().__init__()
        self.num_views = num_views
        self.alpha = nn.Parameter(torch.zeros(num_views))

    def forward(self, logits_list):
        w = F.softmax(self.alpha, dim=0)                 # [num_views]
        stacked = torch.stack(logits_list, dim=0)        # [num_views, B, C]
        return (w.view(-1, 1, 1) * stacked).sum(dim=0)   # [B, C]


# ===========================================================================
# REGISTRY + FACTORIES
# ===========================================================================

# Feature-level fusion methods (use FusionClassifier wrapper)
FEATURE_FUSION_REGISTRY = {
    "concat": ConcatFusion,
    "addweighted": AddWeightedFusion,
    "crossattn": CrossAttentionFusion,
    "gmu": GMUFusion,
}

# Late fusion methods (use LateFusionClassifier wrapper)
LATE_FUSION_REGISTRY = {
    "late_avg": LateAvgFusion,
    "late_weighted": LateWeightedFusion,
}

ALL_METHODS = list(FEATURE_FUSION_REGISTRY.keys()) + list(LATE_FUSION_REGISTRY.keys())


def is_late_fusion(method: str) -> bool:
    return method.lower() in LATE_FUSION_REGISTRY


def build_fusion(method: str, dim: int = 1024, num_views: int = 2, **kwargs):
    method = method.lower()
    if method in FEATURE_FUSION_REGISTRY:
        return FEATURE_FUSION_REGISTRY[method](dim=dim, num_views=num_views, **kwargs)
    if method in LATE_FUSION_REGISTRY:
        return LATE_FUSION_REGISTRY[method](num_views=num_views)
    raise ValueError(f"Unknown fusion method '{method}'. Available: {ALL_METHODS}")


# ===========================================================================
# WRAPPERS: feature-level and late-level full classifiers
# ===========================================================================

class FusionClassifier(nn.Module):
    """Feature-level fusion: features -> fusion -> head -> logits."""

    def __init__(self, method: str, num_classes: int = 199, dim: int = 1024,
                 num_views: int = 2, hidden_dim: int = None,
                 dropout: float = 0.3, fusion_kwargs: dict = None):
        super().__init__()
        fusion_kwargs = fusion_kwargs or {}
        self.fusion = build_fusion(method, dim=dim, num_views=num_views, **fusion_kwargs)
        out_dim = self.fusion.out_dim
        if hidden_dim is None:
            hidden_dim = out_dim
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, views):
        fused = self.fusion(views)
        return self.head(fused)


class LateFusionClassifier(nn.Module):
    """Late fusion: each view -> its own head -> fuse logits.

    Each view has an independent MLP classifier identical in structure to
    FusionClassifier's head (LN -> Linear(D, D) -> GELU -> Linear(D, C)).
    """

    def __init__(self, method: str, num_classes: int = 199, dim: int = 1024,
                 num_views: int = 2, hidden_dim: int = None, dropout: float = 0.3):
        super().__init__()
        if not is_late_fusion(method):
            raise ValueError(f"'{method}' is not a late fusion method")
        if hidden_dim is None:
            hidden_dim = dim

        self.num_views = num_views
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Dropout(dropout),
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
            for _ in range(num_views)
        ])
        self.fusion = build_fusion(method, num_views=num_views)

        # Compatibility with FusionClassifier-style result.json
        self.out_dim = num_classes  # logits dim after fusion

    def forward(self, views):
        # views: list of [B, D]
        per_view_logits = [self.heads[i](views[i]) for i in range(self.num_views)]
        return self.fusion(per_view_logits)


def build_classifier(method: str, num_classes: int = 199, dim: int = 1024,
                     num_views: int = 2, hidden_dim: int = None,
                     dropout: float = 0.3, fusion_kwargs: dict = None):
    """Unified factory: returns the right wrapper based on method type."""
    if is_late_fusion(method):
        return LateFusionClassifier(method=method, num_classes=num_classes,
                                    dim=dim, num_views=num_views,
                                    hidden_dim=hidden_dim, dropout=dropout)
    return FusionClassifier(method=method, num_classes=num_classes, dim=dim,
                            num_views=num_views, hidden_dim=hidden_dim,
                            dropout=dropout, fusion_kwargs=fusion_kwargs)


# ===========================================================================
# Self-test: 6 methods x 2 num_views = 12 cases
# ===========================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    B, D = 4, 1024
    NUM_CLASSES = 199

    expected_out_dim = {
        # feature-level
        ("concat", 2): 2048, ("concat", 3): 3072,
        ("addweighted", 2): 1024, ("addweighted", 3): 1024,
        ("crossattn", 2): 2048, ("crossattn", 3): 3072,
        ("gmu", 2): 1024, ("gmu", 3): 1024,
        # late: out_dim attr stores logits dim
        ("late_avg", 2): NUM_CLASSES, ("late_avg", 3): NUM_CLASSES,
        ("late_weighted", 2): NUM_CLASSES, ("late_weighted", 3): NUM_CLASSES,
    }

    print(f"{'method':<16}{'views':<7}{'out_dim':<10}{'logits':<14}{'#params':<12}{'type':<10}status")
    print("-" * 80)

    all_pass = True
    for method in ALL_METHODS:
        for nv in [2, 3]:
            try:
                views = [torch.randn(B, D) for _ in range(nv)]
                model = build_classifier(method=method, num_classes=NUM_CLASSES,
                                         dim=D, num_views=nv, dropout=0.1)
                logits = model(views)
                n_params = sum(p.numel() for p in model.parameters())

                # For feature methods, out_dim is on .fusion; for late, on model
                if is_late_fusion(method):
                    actual_out_dim = model.out_dim
                    mtype = "late"
                else:
                    actual_out_dim = model.fusion.out_dim
                    mtype = "feat"

                ok_out = actual_out_dim == expected_out_dim[(method, nv)]
                ok_logits = logits.shape == (B, NUM_CLASSES)
                ok_finite = torch.isfinite(logits).all().item()

                loss = logits.sum()
                loss.backward()

                status = "OK" if (ok_out and ok_logits and ok_finite) else "FAIL"
                if status != "OK":
                    all_pass = False
                print(f"{method:<16}{nv:<7}{actual_out_dim:<10}{str(tuple(logits.shape)):<14}"
                      f"{n_params/1e6:<10.2f}M  {mtype:<10}{status}")
            except Exception as e:
                all_pass = False
                print(f"{method:<16}{nv:<7}ERROR: {e}")

    print("-" * 80)
    print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")
