"""
CBAM modules cho VTN3GCN.

Bao gồm 3 tầng attention:
  - Tầng 1: STCBAM_AttnPool  (Spatio-Temporal CBAM + Attention Pooling)
            Áp BÊN TRONG FeatureExtractor, thay AvgPool truyền thống.
  - Tầng 2: CSMAC            (Cross-Stream Modality Attention with CBAM)
            Áp trong forward_features, sau cat các stream.
  - Tầng 3: IVHF             (Inter-View Hierarchical Fusion)
            Áp trong VTN3GCN.forward, trước classifier.

Author: ý tưởng đề xuất bởi Claude theo yêu cầu Bui Viet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
#                   TẦNG 1: ST-CBAM + ATTENTION POOLING
# =============================================================================

class ChannelAttention2D(nn.Module):
    """Channel Attention chuẩn CBAM gốc (Woo et al. 2018)."""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        hidden = max(in_channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels)
        )

    def forward(self, x):
        # x: (B*T, C, H, W)
        b, c, h, w = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx  = F.adaptive_max_pool2d(x, 1).view(b, c)
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx))     # (B*T, C)
        return x * attn.view(b, c, 1, 1)


class SpatialAttention2D(nn.Module):
    """Spatial Attention chuẩn CBAM gốc (Conv 7x7)."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        # x: (B*T, C, H, W)
        avg = x.mean(dim=1, keepdim=True)              # (B*T, 1, H, W)
        mx, _ = x.max(dim=1, keepdim=True)             # (B*T, 1, H, W)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class TemporalAttention(nn.Module):
    """Temporal Attention — học frame nào quan trọng trong T frames."""
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        

    def forward(self, x_btchw, batch_size, t):
        # x_btchw: (B*T, C, H, W) -> tính temporal weight cho T
        bt, c, h, w = x_btchw.shape
        x = x_btchw.view(batch_size, t, c, h, w)        # (B, T, C, H, W)
        avg = x.mean(dim=[2, 3, 4])                     # (B, T)
        mx, _ = x.flatten(2).max(dim=2)                 # (B, T)
        feat = torch.stack([avg, mx], dim=1)            # (B, 2, T)
        attn = torch.sigmoid(self.conv(feat)).squeeze(1)  # (B, T)
        x = x * attn.view(batch_size, t, 1, 1, 1)
        return x.view(bt, c, h, w)


class AttentionPool2D(nn.Module):
    """
    Thay AvgPool2d.
    Học cách pool spatial map (H*W tokens) thành 1 vector qua attention.
    Giữ tối đa thông tin spatial thay vì pool đều.
    """
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert in_channels % num_heads == 0, "in_channels phải chia hết cho num_heads"

        self.query = nn.Parameter(torch.randn(1, 1, in_channels) * 0.02)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)
        self.scale = self.head_dim ** -0.5
        # CBAM_T1 fix: zero-init out_proj de epoch 0, cbam_out ~= 0
        # -> tranh death spiral, cho phep cbam_pool hoc gradient an toan
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        # x: (B*T, C, H, W) -> (B*T, C)
        bt, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)           # (B*T, H*W, C)

        q = self.query.expand(bt, -1, -1)               # (B*T, 1, C)
        k = self.k_proj(tokens)                         # (B*T, H*W, C)
        v = self.v_proj(tokens)                         # (B*T, H*W, C)

        # Multi-head reshape
        def reshape_heads(t):
            return t.view(bt, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q = reshape_heads(q)                            # (B*T, nh, 1, hd)
        k = reshape_heads(k)                            # (B*T, nh, H*W, hd)
        v = reshape_heads(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B*T, nh, 1, H*W)
        attn = attn.softmax(dim=-1)
        out = (attn @ v)                                # (B*T, nh, 1, hd)
        out = out.transpose(1, 2).reshape(bt, 1, c)     # (B*T, 1, C)
        out = self.out_proj(out).squeeze(1)             # (B*T, C)
        return out


class STCBAM_AttnPool(nn.Module):
    """
    TẦNG 1: Spatio-Temporal CBAM + Attention Pooling.
    Thay thế cho avg_pool truyền thống trong FeatureExtractor.

    Input:  (B*T, C, H, W)
    Output: (B*T, C)  — đã pool về 1 vector qua attention thay vì AvgPool đều.
    """
    def __init__(self, in_channels, batch_size_hint=None, t_hint=None,
                 reduction=16, spatial_kernel=7, temporal_kernel=3,
                 pool_heads=4):
        super().__init__()
        self.channel_attn  = ChannelAttention2D(in_channels, reduction)
        self.temporal_attn = TemporalAttention(temporal_kernel)
        self.spatial_attn  = SpatialAttention2D(spatial_kernel)
        self.attn_pool     = AttentionPool2D(in_channels, pool_heads)

    def forward(self, x, batch_size, t):
        """
        x: (B*T, C, H, W)
        batch_size, t: cần truyền để tính temporal attention
        """
        residual = x
        x = self.channel_attn(x)
        x = self.temporal_attn(x, batch_size, t)
        x = self.spatial_attn(x)
        x = x + residual                              # residual giữ ổn định
        x = self.attn_pool(x)                          # (B*T, C)
        return x


# =============================================================================
#                   TẦNG 2: CS-MAC (Cross-Stream Modality Attention)
# =============================================================================

class CSMAC(nn.Module):
    """
    Cross-Stream Modality Attention with CBAM.

    Áp lên feature đã cat từ nhiều stream khác nhau (RGB, AGCN, PoseFlow).
    Học weight cho từng stream + temporal attention 1D xuyên streams.

    Input:  (B, T, total_dim)  với total_dim = sum(stream_dims)
    Output: (B, T, total_dim)
    """
    def __init__(self, stream_dims, common_dim=256, num_heads=4):
        """
        stream_dims: list dim mỗi stream, ví dụ:
            [1024, 256, 106]  cho VTNHCPF_GCN (RGB+AGCN+PF)
            [1024, 106]       cho VTNHCPF (chỉ RGB+PF)
        common_dim: dim chung sau pre-projection để cân bằng các stream
        """
        super().__init__()
        self.stream_dims = stream_dims
        self.num_streams = len(stream_dims)
        self.common_dim = common_dim
        self.total_dim = sum(stream_dims)

        # Pre-projection từng stream về common_dim
        self.pre_projs = nn.ModuleList([
            nn.Linear(d, common_dim) for d in stream_dims
        ])

        # Modality channel gate: học weight cho từng stream
        self.modality_gate = nn.Sequential(
            nn.Linear(self.num_streams, self.num_streams * 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_streams * 4, self.num_streams)
        )

        # Temporal self-attention sau khi gate
        attn_dim = self.num_streams * common_dim
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=num_heads, batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(attn_dim)

        # Project về dim gốc để giữ tương thích với bottle_mm phía sau
        self.out_proj = nn.Linear(attn_dim, self.total_dim)

    def forward(self, zp):
        # zp: (B, T, total_dim)
        B, T, _ = zp.shape
        residual = zp

        # 1) Split theo modality + pre-project về common_dim
        splits = torch.split(zp, self.stream_dims, dim=-1)   # tuple of (B, T, d_i)
        projected = [proj(s) for proj, s in zip(self.pre_projs, splits)]
        # Stack thành (B, T, num_streams, common_dim)
        S = torch.stack(projected, dim=2)

        # 2) Modality channel gate
        g = S.mean(dim=[1, 3])                               # (B, num_streams)
        w = torch.softmax(self.modality_gate(g), dim=-1)     # (B, num_streams)
        S_gated = S * w.view(B, 1, self.num_streams, 1)      # broadcast

        # 3) Temporal self-attention xuyên streams
        S_flat = S_gated.view(B, T, self.num_streams * self.common_dim)
        S_attn, _ = self.temporal_attn(S_flat, S_flat, S_flat)
        S_attn = self.temporal_norm(S_attn + S_flat)

        # 4) Project về dim gốc + residual
        out = self.out_proj(S_attn)
        return out + residual


# =============================================================================
#                   TẦNG 3: IV-HF (Inter-View Hierarchical Fusion)
# =============================================================================

class IVHF(nn.Module):
    """
    Inter-View Hierarchical Fusion với CBAM-style attention.

    Input: 3 tensor (B, view_dim) cho left, center, right
    Output: (B, 3 * view_dim) — đã reweight theo view importance
    """
    def __init__(self, view_dim, reduction=16,
                 init_center_bias=0.5, consensus_alpha=0.1):
        """
        init_center_bias: bias init cho center view (vì center critical theo paper VTN3GCN)
        consensus_alpha: hệ số boost theo cross-view consensus
        """
        super().__init__()
        self.view_dim = view_dim
        self.consensus_alpha = consensus_alpha

        hidden = max(view_dim // reduction, 16)
        # Shared MLP cho avg & max pool
        self.mlp = nn.Sequential(
            nn.Linear(view_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)              # output 1 scalar per view
        )

        # Init bias để center view có lợi thế ban đầu
        # bias = [0, init_center_bias, 0] cho [left, center, right]
        self.center_bias = nn.Parameter(
            torch.tensor([0.0, init_center_bias, 0.0]),
            requires_grad=True
        )

    def forward(self, left_ft, center_ft, right_ft):
        # *_ft: (B, view_dim)
        B = left_ft.shape[0]

        # Stack 3 view: (B, 3, view_dim)
        V = torch.stack([left_ft, center_ft, right_ft], dim=1)

        # 1) View-level attention (CBAM-style với avg + max)
        # AvgPool và MaxPool trên channel dim cho mỗi view
        avg_pool = V.mean(dim=-1)                  # (B, 3)  — không dùng trực tiếp,
        # nhưng MLP chạy trên feature: lấy MLP của từng view feature
        # Dùng feature trực tiếp qua MLP
        scores = self.mlp(V).squeeze(-1)           # (B, 3)
        # Cộng bias center
        scores = scores + self.center_bias.view(1, 3)
        alpha = torch.sigmoid(scores)              # (B, 3) — Sigmoid không phải Softmax

        V_gated = V * alpha.unsqueeze(-1)          # (B, 3, view_dim)

        # 2) Cross-view consistency: cosine similarity giữa các view
        l_n = F.normalize(left_ft, dim=-1)
        c_n = F.normalize(center_ft, dim=-1)
        r_n = F.normalize(right_ft, dim=-1)
        sim_lc = (l_n * c_n).sum(dim=-1)
        sim_cr = (c_n * r_n).sum(dim=-1)
        sim_lr = (l_n * r_n).sum(dim=-1)
        consensus = (sim_lc + sim_cr + sim_lr) / 3.0    # (B,)

        # Boost theo consensus
        V_final = V_gated * (1.0 + self.consensus_alpha * consensus.view(B, 1, 1))

        # Concat lại thành (B, 3 * view_dim)
        return V_final.flatten(1)


# =============================================================================
#                            UNIT TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST CBAM MODULES")
    print("=" * 60)

    # Test 1: ST-CBAM + AttnPool
    print("\n[Test 1] STCBAM_AttnPool")
    B, T, C, H, W = 2, 16, 512, 7, 7
    x = torch.randn(B * T, C, H, W)
    module = STCBAM_AttnPool(in_channels=C)
    out = module(x, batch_size=B, t=T)
    print(f"  Input  shape: {tuple(x.shape)}")
    print(f"  Output shape: {tuple(out.shape)}  (expected: ({B*T}, {C}))")
    assert out.shape == (B * T, C)
    print("  ✓ PASS")

    # Test 2: CSMAC với 3 stream (cho VTNHCPF_GCN)
    print("\n[Test 2] CSMAC — 3 streams (VTNHCPF_GCN)")
    B, T = 2, 16
    stream_dims = [1024, 256, 106]
    zp = torch.randn(B, T, sum(stream_dims))
    module = CSMAC(stream_dims=stream_dims, common_dim=256)
    out = module(zp)
    print(f"  Input  shape: {tuple(zp.shape)}")
    print(f"  Output shape: {tuple(out.shape)}  (expected: ({B}, {T}, {sum(stream_dims)}))")
    assert out.shape == zp.shape
    print("  ✓ PASS")

    # Test 3: CSMAC với 2 stream (cho VTNHCPF)
    print("\n[Test 3] CSMAC — 2 streams (VTNHCPF)")
    stream_dims = [1024, 106]
    zp = torch.randn(B, T, sum(stream_dims))
    module = CSMAC(stream_dims=stream_dims, common_dim=256)
    out = module(zp)
    print(f"  Input  shape: {tuple(zp.shape)}")
    print(f"  Output shape: {tuple(out.shape)}  (expected: ({B}, {T}, {sum(stream_dims)}))")
    assert out.shape == zp.shape
    print("  ✓ PASS")

    # Test 4: IVHF
    print("\n[Test 4] IVHF")
    view_dim = 1024
    left  = torch.randn(B, view_dim)
    center = torch.randn(B, view_dim)
    right = torch.randn(B, view_dim)
    module = IVHF(view_dim=view_dim)
    out = module(left, center, right)
    print(f"  Input each view: {tuple(left.shape)}")
    print(f"  Output shape: {tuple(out.shape)}  (expected: ({B}, {3 * view_dim}))")
    assert out.shape == (B, 3 * view_dim)
    print("  ✓ PASS")

    # Test 5: Đếm params
    print("\n[Test 5] Param count")
    m1 = STCBAM_AttnPool(in_channels=512)
    m2 = CSMAC(stream_dims=[1024, 256, 106])
    m3 = IVHF(view_dim=1024)
    p1 = sum(p.numel() for p in m1.parameters())
    p2 = sum(p.numel() for p in m2.parameters())
    p3 = sum(p.numel() for p in m3.parameters())
    print(f"  STCBAM_AttnPool: {p1:,} params")
    print(f"  CSMAC (3-stream): {p2:,} params")
    print(f"  IVHF: {p3:,} params")
    print(f"  Tổng (rough estimate, không tính nhân số instance): {p1+p2+p3:,}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)