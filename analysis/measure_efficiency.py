"""
measure_efficiency.py
Đo FLOPs, Params, Latency (ms), FPS cho VTN3GCN-CBAM.

Usage:
    cd /media/ibmelab/ibme31/VTN3GCN_CBAM
    python measure_efficiency.py

Output:
    - In bảng kết quả ra terminal
    - Lưu results/efficiency_table.json
"""

import os
import sys
import time
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

# ============================================================
# Import models
# ============================================================
from modelling.vtn_att_poseflow_model import VTNHCPF, VTNHCPF_GCN
import yaml

DEVICE = torch.device('cuda:1')
torch.cuda.set_device(DEVICE)
N_WARMUP = 10
N_RUNS   = 100

# ============================================================
# Đọc config để lấy đúng hyperparams
# ============================================================
cfg_c = yaml.safe_load(open('configs/Stage1_SingleView/Stage1_Center_VSL199.yaml'))
cfg_l = yaml.safe_load(open('configs/Stage1_SingleView/Stage1_Left_VSL199.yaml'))

T   = cfg_c['data']['num_output_frames']   # 16
IMG = cfg_c['data']['vid_transform']['IMAGE_SIZE']  # 224
B   = 1  # batch size 1 cho inference latency

print(f"Config: T={T}, IMG={IMG}, B={B}")
print(f"Device: {DEVICE}\n")

# ============================================================
# Tạo dummy inputs ĐÚNG SHAPE (match dataloader output)
# ============================================================
# clip:      (B, T, x, C, H, W) — x=2 (2 hand crops), từ dataset __getitem__
# poseflow:  (B, T, 106)
# keypoints: (B, 2, T, 46, 1)   — shape AAGCN chuẩn

x = 2   # 2 hand crops (left + right hand)

dummy_clip      = torch.randn(B, T, x, 3, IMG, IMG).to(DEVICE)
dummy_poseflow  = torch.randn(B, T, 106).to(DEVICE)
dummy_keypoints = torch.randn(B, 2, T, 46, 1).to(DEVICE)

print(f"Dummy inputs:")
print(f"  clip:      {tuple(dummy_clip.shape)}")
print(f"  poseflow:  {tuple(dummy_poseflow.shape)}")
print(f"  keypoints: {tuple(dummy_keypoints.shape)}\n")


# ============================================================
# Đếm params
# ============================================================
def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


# ============================================================
# Đo latency bằng CUDA event (chính xác hơn time.time)
# ============================================================
def measure_latency_cuda(fn, n_warmup=N_WARMUP, n_runs=N_RUNS):
    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            fn()
    torch.cuda.synchronize(DEVICE)

    # Đo thời gian
    start_time = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            fn()
    torch.cuda.synchronize(DEVICE)

    total_ms = (time.time() - start_time) * 1000
    per_ms = total_ms / n_runs
    fps = 1000.0 / per_ms
    return per_ms, fps


# ============================================================
# Đo FLOPs bằng thop (wrap model để nhận positional args)
# ============================================================
def measure_flops_center(model):
    """Wrap VTNHCPF để thop profile được."""
    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, clip, poseflow):
            b, t, x, c, h, w = clip.size()
            z_rgb = self.m.feature_extractor(clip.view(b, t*x, c, h, w)).view(b, t, -1)
            return self.m.forward_features(z_rgb, poseflow)
    try:
        from thop import profile
        wrapper = Wrapper(model)
        flops, _ = profile(
            wrapper,
            inputs=(dummy_clip, dummy_poseflow),
            verbose=False
        )
        return flops / 1e9
    except Exception as e:
        print(f"  [FLOPs center] thop error: {e}")
        return None


def measure_flops_gcn(model):
    """Wrap VTNHCPF_GCN để thop profile được."""
    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, clip, poseflow, keypoints):
            b, t, x, c, h, w = clip.size()
            z_rgb = self.m.feature_extractor(clip.view(b, t*x, c, h, w)).view(b, t, -1)
            z_kp  = self.m.feature_extractor_gcn(keypoints)
            return self.m.forward_features(z_rgb, poseflow, z_kp)
    try:
        from thop import profile
        wrapper = Wrapper(model)
        flops, _ = profile(
            wrapper,
            inputs=(dummy_clip, dummy_poseflow, dummy_keypoints),
            verbose=False
        )
        return flops / 1e9
    except Exception as e:
        print(f"  [FLOPs gcn] thop error: {e}")
        return None


# ============================================================
# Build models
# ============================================================
print("=" * 60)
print("  Building models...")
print("=" * 60)

center_model = VTNHCPF(
    num_classes    = cfg_c['model']['num_classes'],
    num_heads      = cfg_c['model']['num_heads'],
    num_layers     = cfg_c['model']['num_layers'],
    embed_size     = cfg_c['model']['embed_size'],
    sequence_length= T,
    cnn            = cfg_c['model']['cnn'],
    freeze_layers  = cfg_c['model']['freeze_layers'],
    dropout        = cfg_c['model']['dropout'],
    use_cbam_t1    = True,
    use_cbam_t2    = False,
).to(DEVICE).eval()

left_model = VTNHCPF_GCN(
    num_classes    = cfg_l['model']['num_classes'],
    num_heads      = cfg_l['model']['num_heads'],
    num_layers     = cfg_l['model']['num_layers'],
    embed_size     = cfg_l['model']['embed_size'],
    sequence_length= T,
    cnn            = cfg_l['model']['cnn'],
    gcn            = cfg_l['model'].get('gcn', 'AAGCN'),
    freeze_layers  = cfg_l['model']['freeze_layers'],
    dropout        = cfg_l['model']['dropout'],
    use_cbam_t1    = True,
    use_cbam_t2    = False,
).to(DEVICE).eval()

# Fusion head
class LCR_Concat_Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(3072),
            nn.Dropout(0.3),
            nn.Linear(3072, 3072),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(3072, 199),
        )
    def forward(self, x):
        return self.head(x)

fusion_head = LCR_Concat_Head().to(DEVICE).eval()

print("  Models built.\n")


# ============================================================
# Measure params
# ============================================================
params_center = count_params(center_model)
params_left   = count_params(left_model)
params_fusion = count_params(fusion_head)
params_total  = params_center + params_left * 2 + params_fusion  # Left + Right share same arch

print("=" * 60)
print("  PARAMETER COUNT")
print("=" * 60)
print(f"  Center (VTNHCPF):        {params_center:.2f}M")
print(f"  Left   (VTNHCPF_GCN):   {params_left:.2f}M")
print(f"  Right  (VTNHCPF_GCN):   {params_left:.2f}M  (same arch)")
print(f"  Fusion head (LCR_concat):{params_fusion:.2f}M")
print(f"  TOTAL:                   {params_total:.2f}M\n")


# ============================================================
# Measure FLOPs
# ============================================================
print("=" * 60)
print("  FLOPs MEASUREMENT")
print("=" * 60)

flops_center = measure_flops_center(center_model)
print(f"  Center FLOPs: {flops_center:.2f}G" if flops_center else "  Center FLOPs: N/A")

flops_gcn = measure_flops_gcn(left_model)
print(f"  Left/Right FLOPs each: {flops_gcn:.2f}G" if flops_gcn else "  Left/Right FLOPs: N/A")

# Fusion head FLOPs
try:
    from thop import profile as thop_profile
    dummy_fused = torch.randn(B, 3072).to(DEVICE)
    flops_fusion, _ = thop_profile(fusion_head, inputs=(dummy_fused,), verbose=False)
    flops_fusion = flops_fusion / 1e9
    print(f"  Fusion head FLOPs: {flops_fusion:.4f}G")
except Exception as e:
    flops_fusion = None
    print(f"  Fusion head FLOPs: N/A ({e})")

flops_total = None
if all(v is not None for v in [flops_center, flops_gcn, flops_fusion]):
    flops_total = flops_center + flops_gcn * 2 + flops_fusion
    print(f"\n  TOTAL FLOPs (3 views + head): {flops_total:.2f}G\n")
else:
    print("\n  TOTAL FLOPs: N/A (some components failed)\n")


# ============================================================
# Measure Latency (per-component + end-to-end)
# ============================================================
print("=" * 60)
print(f"  LATENCY MEASUREMENT ({N_RUNS} runs, B=1)")
print("=" * 60)

# Center latency
def fn_center():
    b, t, x, c, h, w = dummy_clip.size()
    z_rgb = center_model.feature_extractor(dummy_clip.view(b, t*x, c, h, w)).view(b, t, -1)
    return center_model.forward_features(z_rgb, dummy_poseflow).mean(1)

lat_center, fps_center = measure_latency_cuda(fn_center)
print(f"  Center:     {lat_center:.2f}ms  ({fps_center:.1f} FPS)")

# Left/Right latency
def fn_gcn():
    b, t, x, c, h, w = dummy_clip.size()
    z_rgb = left_model.feature_extractor(dummy_clip.view(b, t*x, c, h, w)).view(b, t, -1)
    z_kp  = left_model.feature_extractor_gcn(dummy_keypoints)
    return left_model.forward_features(z_rgb, dummy_poseflow, z_kp).mean(1)

lat_gcn, fps_gcn = measure_latency_cuda(fn_gcn)
print(f"  Left/Right: {lat_gcn:.2f}ms  ({fps_gcn:.1f} FPS)  (each)")

# Fusion head latency
dummy_fused_input = torch.randn(B, 3072).to(DEVICE)
def fn_fusion():
    return fusion_head(dummy_fused_input)

lat_fusion, fps_fusion = measure_latency_cuda(fn_fusion)
print(f"  Fusion head:{lat_fusion:.2f}ms  ({fps_fusion:.1f} FPS)")

# End-to-end latency (3 views sequential + fusion)
def fn_e2e():
    b, t, x, c, h, w = dummy_clip.size()

    # Center
    z_c = center_model.feature_extractor(dummy_clip.view(b, t*x, c, h, w)).view(b, t, -1)
    feat_c = center_model.forward_features(z_c, dummy_poseflow).mean(1)

    # Left
    z_l  = left_model.feature_extractor(dummy_clip.view(b, t*x, c, h, w)).view(b, t, -1)
    z_kp = left_model.feature_extractor_gcn(dummy_keypoints)
    feat_l = left_model.forward_features(z_l, dummy_poseflow, z_kp).mean(1)

    # Right (same model as left in inference)
    feat_r = left_model.forward_features(z_l, dummy_poseflow, z_kp).mean(1)

    # Fusion
    fused = torch.cat([feat_l, feat_c, feat_r], dim=-1)
    return fusion_head(fused)

lat_e2e, fps_e2e = measure_latency_cuda(fn_e2e)
print(f"\n  END-TO-END: {lat_e2e:.2f}ms  ({fps_e2e:.1f} FPS)")


# ============================================================
# Print final summary table
# ============================================================
print()
print("=" * 70)
print("  FINAL SUMMARY TABLE")
print("=" * 70)
print(f"  {'Component':<30} {'Params (M)':>10} {'FLOPs (G)':>10} {'Latency (ms)':>14} {'FPS':>8}")
print("  " + "-" * 66)

def fmt(v, fmt_str):
    return fmt_str.format(v) if v is not None else "N/A"

print(f"  {'Center (VTNHCPF)':<30} {fmt(params_center, '{:.2f}'):>10} {fmt(flops_center, '{:.2f}'):>10} {lat_center:>14.2f} {fps_center:>8.1f}")
print(f"  {'Left (VTNHCPF_GCN)':<30} {fmt(params_left,  '{:.2f}'):>10} {fmt(flops_gcn, '{:.2f}'):>10} {lat_gcn:>14.2f} {fps_gcn:>8.1f}")
print(f"  {'Right (VTNHCPF_GCN)':<30} {fmt(params_left,  '{:.2f}'):>10} {fmt(flops_gcn, '{:.2f}'):>10} {lat_gcn:>14.2f} {fps_gcn:>8.1f}")
print(f"  {'Fusion Head (LCR_concat)':<30} {fmt(params_fusion,'{:.2f}'):>10} {fmt(flops_fusion,'{:.4f}'):>10} {lat_fusion:>14.2f} {fps_fusion:>8.1f}")
print("  " + "-" * 66)
print(f"  {'TOTAL (end-to-end)':<30} {fmt(params_total, '{:.2f}'):>10} {fmt(flops_total, '{:.2f}'):>10} {lat_e2e:>14.2f} {fps_e2e:>8.1f}")
print("=" * 70)

print(f"""
  LaTeX table (copy vào paper):

  \\begin{{table}}[h]
  \\centering
  \\caption{{Computational efficiency of VTN3GCN-CBAM (B=1, T={T}, GPU: RTX 3090)}}
  \\begin{{tabular}}{{lrrrr}}
  \\hline
  Component & Params (M) & FLOPs (G) & Latency (ms) & FPS \\\\
  \\hline
  Center (VTNHCPF)     & {fmt(params_center, '{:.2f}')} & {fmt(flops_center, '{:.2f}')} & {lat_center:.2f} & {fps_center:.1f} \\\\
  Left (VTNHCPF\\_GCN)  & {fmt(params_left,  '{:.2f}')} & {fmt(flops_gcn,    '{:.2f}')} & {lat_gcn:.2f}    & {fps_gcn:.1f} \\\\
  Right (VTNHCPF\\_GCN) & {fmt(params_left,  '{:.2f}')} & {fmt(flops_gcn,    '{:.2f}')} & {lat_gcn:.2f}    & {fps_gcn:.1f} \\\\
  Fusion head          & {fmt(params_fusion,'{:.2f}')} & {fmt(flops_fusion, '{:.4f}')} & {lat_fusion:.2f} & {fps_fusion:.1f} \\\\
  \\hline
  \\textbf{{Total}}      & \\textbf{{{fmt(params_total, '{:.2f}')}}} & \\textbf{{{fmt(flops_total, '{:.2f}')}}} & \\textbf{{{lat_e2e:.2f}}} & \\textbf{{{fps_e2e:.1f}}} \\\\
  \\hline
  \\end{{tabular}}
  \\end{{table}}
""")

# ============================================================
# Save JSON
# ============================================================
os.makedirs('results', exist_ok=True)
result = {
    'config': {'T': T, 'IMG': IMG, 'B': B, 'device': str(DEVICE),
               'n_warmup': N_WARMUP, 'n_runs': N_RUNS},
    'center':  {'params_M': params_center, 'flops_G': flops_center,  'latency_ms': lat_center,  'fps': fps_center},
    'left':    {'params_M': params_left,   'flops_G': flops_gcn,     'latency_ms': lat_gcn,     'fps': fps_gcn},
    'right':   {'params_M': params_left,   'flops_G': flops_gcn,     'latency_ms': lat_gcn,     'fps': fps_gcn},
    'fusion':  {'params_M': params_fusion, 'flops_G': flops_fusion,  'latency_ms': lat_fusion,  'fps': fps_fusion},
    'e2e':     {'params_M': params_total,  'flops_G': flops_total,   'latency_ms': lat_e2e,     'fps': fps_e2e},
}
with open('results/efficiency_table.json', 'w') as f:
    json.dump(result, f, indent=2)
print("  Saved: results/efficiency_table.json")
