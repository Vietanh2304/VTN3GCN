import torch
from torch import nn

from .vtn_utils import FeatureExtractor, FeatureExtractorGCN, LinearClassifier, SelfAttention, CrossAttention
from .cbam_modules import CSMAC, IVHF
import torch.nn.functional as F
from pytorch_lightning.utilities.migration import pl_legacy_patch

def _ramped_md_prob(current_epoch: int,
                    warmup: int = 5,
                    ramp: int = 10,
                    max_prob: float = 0.3) -> float:
    """Modality dropout with warmup + linear ramp.
    [0, warmup):           0.0
    [warmup, warmup+ramp): linear 0 -> max_prob
    [warmup+ramp, inf):    max_prob
    """
    if current_epoch < warmup:
        return 0.0
    if current_epoch < warmup + ramp:
        return (current_epoch - warmup) / ramp * max_prob
    return max_prob
class MMTensorNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

    def forward(self, x):
        mean = torch.mean(x, dim=self.dim).unsqueeze(self.dim)
        std = torch.std(x, dim=self.dim).unsqueeze(self.dim)
        return (x - mean) / std


class VTNHCPF(nn.Module):
    def __init__(self, num_classes=199, num_heads=4, num_layers=2, embed_size=512,
                 sequence_length=16, cnn='rn34', freeze_layers=0, dropout=0,
                 use_cbam_t1=True, use_cbam_t2=True, **kwargs):
        super().__init__()
        print("Model: VTNHCPF")
        self.sequence_length = sequence_length
        self.embed_size = embed_size
        self.num_classes = num_classes
        self.num_attn_features = embed_size * 2     # = 1024 cho HANDCROP
        self.use_cbam_t2 = use_cbam_t2

        self.feature_extractor = FeatureExtractor(cnn, embed_size, freeze_layers,
                                                  use_cbam=use_cbam_t1)
        self.norm = nn.LayerNorm(106 + self.num_attn_features)  # = 1130
        self.bottle_mm = nn.Linear(106 + self.num_attn_features, self.num_attn_features)

        # CS-MAC tầng 2 cho 2-stream [RGB=1024, PF=106]
        if use_cbam_t2:
            self.csmac = CSMAC(stream_dims=[self.num_attn_features, 106],
                       common_dim=256, num_heads=4)
            self.beta_t2 = nn.Parameter(torch.zeros(1))  # ReZero gate

        self.self_attention_decoder = SelfAttention(
            self.num_attn_features, self.num_attn_features,
            [num_heads] * num_layers,
            self.sequence_length, layer_norm=True, dropout=dropout
        )
        self.classifier = LinearClassifier(self.num_attn_features, self.num_classes, dropout)
        self.dropout = dropout
        self.relu = F.relu
        # Ramped modality dropout
        self.register_buffer('md_current_epoch', torch.tensor(0, dtype=torch.long))
        self.md_warmup_epochs = 5
        self.md_ramp_epochs = 10
        self.md_max_prob = 0.30

    def reset_head(self, num_classes):
        self.classifier = LinearClassifier(self.num_attn_features, num_classes, self.dropout)
        print("Reset to ", num_classes)

    def forward_features(self, features=None, poseflow=None):
        if self.training:
            _md_p = _ramped_md_prob(
                int(getattr(self, 'md_current_epoch', torch.tensor(999)).item()),
                warmup=getattr(self, 'md_warmup_epochs', 5),
                ramp=getattr(self, 'md_ramp_epochs', 10),
                max_prob=getattr(self, 'md_max_prob', 0.25),
            )
            if _md_p > 0 and torch.rand(1).item() < _md_p:
                features = torch.zeros_like(features)
        zp = torch.cat((features, poseflow), dim=-1)         # (B, T, 1024+106=1130)

        if self.use_cbam_t2:
            zp = zp + self.beta_t2 * self.csmac(zp)                                 # CS-MAC tầng 2

        zp = self.norm(zp)
        zp = self.relu(self.bottle_mm(zp))
        zp = self.self_attention_decoder(zp)
        return zp

    def forward(self, clip=None, poseflow=None, **kwargs):
        rgb_clip, pose_clip = clip, poseflow
        b, t, x, c, h, w = rgb_clip.size()
        rgb_clip = rgb_clip.view(b, t * x, c, h, w)
        z = self.feature_extractor(rgb_clip)
        z = z.view(b, t, -1)                                  # (B, T, 1024)

        zp = self.forward_features(z, pose_clip)
        y = self.classifier(zp)
        return {'logits': y.mean(1)}


class VTNHCPF_GCN(nn.Module):
    def __init__(self, num_classes=199, num_heads=4, num_layers=2, embed_size=512,
                 sequence_length=16, cnn='rn34', gcn='AAGCN', freeze_layers=0,
                 dropout=0, use_cbam_t1=True, use_cbam_t2=True, **kwargs):
        super().__init__()
        print("Model: VTNHCPF_GCN")
        self.sequence_length = sequence_length
        self.embed_size = embed_size
        self.num_classes = num_classes
        self.use_cbam_t2 = use_cbam_t2

        self.feature_extractor = FeatureExtractor(cnn, embed_size, freeze_layers,
                                                  use_cbam=use_cbam_t1)
        self.feature_extractor_gcn = FeatureExtractorGCN(gcn, freeze_layers)

        num_attn_features = embed_size                         # 512
        num_gcn_features = int(embed_size / 2)                 # 256
        pose_flow_features = 106
        add_attn_features = embed_size                         # 512

        # Concat input dim:
        # RGB(2 hands * embed_size) + AGCN(num_gcn_features) + PF(106)
        # = 1024 + 256 + 106 = 1386  (HANDCROP)
        rgb_dim = num_attn_features * 2
        concat_dim = rgb_dim + num_gcn_features + pose_flow_features
        out_dim = num_attn_features + add_attn_features        # 1024

        self.norm = nn.LayerNorm(concat_dim)  # = 1386
        self.bottle_mm = nn.Linear(concat_dim, out_dim)

        # CS-MAC tầng 2 cho 3-stream [RGB=1024, AGCN=256, PF=106]
        if use_cbam_t2:
            self.csmac = CSMAC(stream_dims=[rgb_dim, num_gcn_features, pose_flow_features],
                       common_dim=256, num_heads=4)
            self.beta_t2 = nn.Parameter(torch.zeros(1))  # ReZero gate

        self.self_attention_decoder = SelfAttention(
            out_dim, out_dim,
            [num_heads] * num_layers,
            self.sequence_length, layer_norm=True, dropout=dropout
        )
        self.classifier = LinearClassifier(out_dim, num_classes, dropout)
        self.num_attn_features = out_dim
        self.dropout = dropout
        self.num_classes = num_classes
        self.relu = F.relu
        # Ramped modality dropout
        self.register_buffer('md_current_epoch', torch.tensor(0, dtype=torch.long))
        self.md_warmup_epochs = 5
        self.md_ramp_epochs = 10
        self.md_max_prob = 0.30

    def reset_head(self, num_classes):
        self.classifier = LinearClassifier(self.num_attn_features, num_classes, self.dropout)
        print("Reset to ", num_classes)

    def forward_features(self, features=None, poseflow=None, features_keypoint=None):
        # cat: [RGB=1024, AGCN=256, PF=106] = 1386
        if self.training:
            _md_p = _ramped_md_prob(
                int(getattr(self, 'md_current_epoch', torch.tensor(999)).item()),
                warmup=getattr(self, 'md_warmup_epochs', 5),
                ramp=getattr(self, 'md_ramp_epochs', 10),
                max_prob=getattr(self, 'md_max_prob', 0.25),
            )
            if _md_p > 0 and torch.rand(1).item() < _md_p:
                features = torch.zeros_like(features)
        zp = torch.cat((features, features_keypoint, poseflow), dim=-1)

        if self.use_cbam_t2:
            zp = zp + self.beta_t2 * self.csmac(zp)                                # CS-MAC tầng 2

        zp = self.norm(zp)
        zp = self.relu(self.bottle_mm(zp))
        zp = self.self_attention_decoder(zp)
        return zp

    def forward(self, clip=None, poseflow=None, keypoints=None, **kwargs):
        rgb_clip, pose_clip = clip, poseflow
        b, t, x, c, h, w = rgb_clip.size()
        rgb_clip = rgb_clip.view(b, t * x, c, h, w)
        zc = self.feature_extractor(rgb_clip)
        zc = zc.view(b, t, -1)                                 # (B, T, 1024)

        zk = self.feature_extractor_gcn(keypoints)             # (B, T, 256)

        zp = self.forward_features(zc, pose_clip, zk)
        y = self.classifier(zp)
        return {'logits': y.mean(1)}


class VTNHCPF_Three_View(nn.Module):
    def __init__(self, num_classes=199, num_heads=4, num_layers=2, embed_size=512, sequence_length=16, cnn='rn34',
                 freeze_layers=0, dropout=0, **kwargs):
        super().__init__()
        print("Model: VTNHCPF_Three_View")
        self.center = VTNHCPF(num_classes,num_heads,num_layers,embed_size,sequence_length,cnn,freeze_layers,dropout,**kwargs)
        self.left = VTNHCPF(num_classes,num_heads,num_layers,embed_size,sequence_length,cnn,freeze_layers,dropout,**kwargs)
        self.right = VTNHCPF(num_classes,num_heads,num_layers,embed_size,sequence_length,cnn,freeze_layers,dropout,**kwargs)
        self.classifier = LinearClassifier(embed_size*2*3, num_classes, dropout)
        self.feature_extractor = None
    def add_backbone(self):
        self.feature_extractor = self.center.feature_extractor

    def remove_head_and_backbone(self):
        self.center.feature_extractor = nn.Identity()
        self.left.feature_extractor = nn.Identity()
        self.right.feature_extractor = nn.Identity()
        self.center.classifier = nn.Identity()
        self.left.classifier = nn.Identity()
        self.right.classifier = nn.Identity()
        print("Remove head and backbone")
    
    def freeze(self,layers = 2):
        print(f"Freeze {layers} layers attn")
        for i in range(layers):
            for param in self.center.self_attention_decoder.layers[i].parameters():
                param.requires_grad = False
            for param in self.left.self_attention_decoder.layers[i].parameters():
                param.requires_grad = False
            for param in self.right.self_attention_decoder.layers[i].parameters():
                param.requires_grad = False
    def forward_features(self,left = None,center = None,right = None,center_pf = None,left_pf = None,right_pf = None): 
        b, t, x, c, h, w = left.size()
        left_feature = self.feature_extractor(left.view(b, t * x, c, h, w)).view(b,t,-1)
        right_feature = self.feature_extractor(right.view(b, t * x, c, h, w)).view(b,t,-1)
        center_feature = self.feature_extractor(center.view(b, t * x, c, h, w)).view(b,t,-1)
        
        left_ft = self.left.forward_features(left_feature,left_pf).mean(1)
        center_ft = self.center.forward_features(center_feature,center_pf).mean(1)
        right_ft = self.right.forward_features(right_feature,right_pf).mean(1)
        
        output_features = torch.cat([left_ft,center_ft,right_ft],dim = -1)
       
        return output_features
    def forward(self,left = None,center = None,right = None,center_pf = None,left_pf = None,right_pf = None):  
        b, t, x, c, h, w = left.size()
        left_feature = self.feature_extractor(left.view(b, t * x, c, h, w)).view(b,t,-1)
        right_feature = self.feature_extractor(right.view(b, t * x, c, h, w)).view(b,t,-1)
        center_feature = self.feature_extractor(center.view(b, t * x, c, h, w)).view(b,t,-1)
        
        left_ft = self.left.forward_features(left_feature,left_pf).mean(1)
        center_ft = self.center.forward_features(center_feature,center_pf).mean(1)
        right_ft = self.right.forward_features(right_feature,right_pf).mean(1)
        
        output_features = torch.cat([left_ft,center_ft,right_ft],dim = -1)
       

        y = self.classifier(output_features)

        return {
            'logits':y
        }

class VTN3GCN(nn.Module):
    def __init__(self, num_classes=199, num_heads=4, num_layers=2, embed_size=512,
                 sequence_length=16, cnn='rn34', gcn='AAGCN', freeze_layers=0,
                 dropout=0, use_cbam_t1=True, use_cbam_t2=True, use_cbam_t3=True,
                 **kwargs):
        super().__init__()
        print("Model: VTN3GCN")
        self.use_cbam_t3 = use_cbam_t3

        # 3 sub-models — share flag T1, T2 xuống dưới
        common_kwargs = dict(
            num_classes=num_classes, num_heads=num_heads, num_layers=num_layers,
            embed_size=embed_size, sequence_length=sequence_length, cnn=cnn,
            freeze_layers=freeze_layers, dropout=dropout,
            use_cbam_t1=use_cbam_t1, use_cbam_t2=use_cbam_t2
        )
        # Không truyền **kwargs xuống sub-models để tránh duplicate keys
        self.center = VTNHCPF(**common_kwargs)
        self.left   = VTNHCPF_GCN(gcn=gcn, **common_kwargs)
        self.right  = VTNHCPF_GCN(gcn=gcn, **common_kwargs)

        # Mỗi view ra dim = embed_size * 2 = 1024
        view_dim = embed_size * 2
        self.view_dim = view_dim

        # FIX BUG: classifier đúng phải là embed_size*6 = 3072 (không phải 1536)
        self.classifier = LinearClassifier(view_dim * 3, num_classes, dropout)

        # Tầng 3: IVHF
        if use_cbam_t3:
            self.ivhf = IVHF(view_dim=view_dim)
            self.gamma_t3 = nn.Parameter(torch.zeros(1))  # ReZero gate

        self.embed_size = embed_size
        self.feature_extractor = None
        self.feature_extractor_gcn_right = None
        self.feature_extractor_gcn_left = None

    def add_backbone(self):
        self.feature_extractor = self.center.feature_extractor
        self.feature_extractor_gcn_right = self.right.feature_extractor_gcn
        self.feature_extractor_gcn_left = self.left.feature_extractor_gcn

    def remove_head_and_backbone(self):
        self.center.feature_extractor = nn.Identity()
        self.left.feature_extractor = nn.Identity()
        self.right.feature_extractor = nn.Identity()
        self.left.feature_extractor_gcn = nn.Identity()
        self.right.feature_extractor_gcn = nn.Identity()
        self.center.classifier = nn.Identity()
        self.left.classifier = nn.Identity()
        self.right.classifier = nn.Identity()
        print("Remove head and backbone")

    def freeze(self, layers=2):
        print(f"Freeze {layers} layers attn")
        for i in range(layers):
            for param in self.center.self_attention_decoder.layers[i].parameters():
                param.requires_grad = False
            for param in self.left.self_attention_decoder.layers[i].parameters():
                param.requires_grad = False
            for param in self.right.self_attention_decoder.layers[i].parameters():
                param.requires_grad = False

    def forward_features(self, left=None, center=None, right=None, left_kp=None, right_kp=None, 
                     center_kp=None, center_pf=None, left_pf=None, right_pf=None): 
        # HANDCROP mode
        b, t, x, c, h, w = left.size()
        
        # Gộp x vào b, giữ nguyên t
        left_in = left.transpose(1, 2).contiguous().view(b * x, t, c, h, w)
        right_in = right.transpose(1, 2).contiguous().view(b * x, t, c, h, w)
        center_in = center.transpose(1, 2).contiguous().view(b * x, t, c, h, w)

        # Đi qua FeatureExtractor (có chứa ST-CBAM)
        left_feature = self.feature_extractor(left_in)
        right_feature = self.feature_extractor(right_in)
        center_feature = self.feature_extractor(center_in)

        # Tách x ra và gộp vào dim cuối
        left_feature = left_feature.view(b, x, t, -1).transpose(1, 2).contiguous().view(b, t, -1)
        right_feature = right_feature.view(b, x, t, -1).transpose(1, 2).contiguous().view(b, t, -1)
        center_feature = center_feature.view(b, x, t, -1).transpose(1, 2).contiguous().view(b, t, -1)

        left_kp_feature = self.feature_extractor_gcn_left(left_kp)
        right_kp_feature = self.feature_extractor_gcn_right(right_kp)
        
        left_ft = self.left.forward_features(left_feature, left_kp_feature, left_pf).mean(1)
        center_ft = self.center.forward_features(center_feature, center_pf).mean(1)
        right_ft = self.right.forward_features(right_feature, right_kp_feature, right_pf).mean(1)
        
        output_features = torch.cat([left_ft, center_ft, right_ft], dim=-1)
        return output_features

    def forward(self, left=None, center=None, right=None,
                left_kp=None, right_kp=None, center_kp=None,
                center_pf=None, left_pf=None, right_pf=None):
        
        # CHẾ ĐỘ HANDCROP 6 CHIỀU
        b, t, x, c, h, w = left.size()

        # 1. BẢO VỆ TRỤC THỜI GIAN: Đảo x lên gộp với b, giữ t nguyên vẹn
        left_in = left.transpose(1, 2).contiguous().view(b * x, t, c, h, w)
        right_in = right.transpose(1, 2).contiguous().view(b * x, t, c, h, w)
        center_in = center.transpose(1, 2).contiguous().view(b * x, t, c, h, w)

        # 2. Đưa qua Feature Extractor (Chứa ST-CBAM)
        left_feature = self.feature_extractor(left_in)
        right_feature = self.feature_extractor(right_in)
        center_feature = self.feature_extractor(center_in)

        # 3. Lật lại đúng shape (b, t, -1) để tương thích với các layer sau
        left_feature = left_feature.view(b, x, t, -1).transpose(1, 2).contiguous().view(b, t, -1)
        right_feature = right_feature.view(b, x, t, -1).transpose(1, 2).contiguous().view(b, t, -1)
        center_feature = center_feature.view(b, x, t, -1).transpose(1, 2).contiguous().view(b, t, -1)

        # 4. Trích xuất Keypoint và Gộp luồng (CSMAC)
        left_kp_feature  = self.feature_extractor_gcn_left(left_kp)
        right_kp_feature = self.feature_extractor_gcn_right(right_kp)

        left_ft   = self.left.forward_features(left_feature, left_kp_feature, left_pf).mean(1)
        center_ft = self.center.forward_features(center_feature, center_pf).mean(1)
        right_ft  = self.right.forward_features(right_feature, right_kp_feature, right_pf).mean(1)

        # 5. Trộn 3 góc nhìn (IVHF Tầng 3)
        output_baseline = torch.cat([left_ft, center_ft, right_ft], dim=-1)  # baseline luôn có
        if self.use_cbam_t3:
            output_features = output_baseline + self.gamma_t3 * self.ivhf(left_ft, center_ft, right_ft)
        else:
            output_features = output_baseline

        # 6. Phân loại
        y = self.classifier(output_features)
        return {'logits': y}
class VTNHCPF_OneView_Sim_Knowledge_Distilation(nn.Module):
    def __init__(self, num_classes=199, num_heads=4, num_layers=2, embed_size=512, sequence_length=16, cnn='rn34',
                 freeze_layers=0, dropout=0, **kwargs):
        super().__init__()
        print("Model: VTNHCPF_OneView_Sim_Knowledge_Distillation")
        self.teacher = VTNHCPF_Three_View(num_classes,num_heads,num_layers,embed_size,sequence_length,cnn,freeze_layers,dropout,**kwargs)
        self.teacher.add_backbone()
        self.teacher.remove_head_and_backbone()
        self.teacher.load_state_dict(torch.load("checkpoints/VTNHCPF_Three_view/vtn_att_poseflow three view finetune from one view with testings labels/best_checkpoints.pth",map_location='cpu'))
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        self.student = VTNHCPF(num_classes,num_heads,num_layers,embed_size,sequence_length,cnn,freeze_layers,dropout,**kwargs)
        new_state_dict = {}
        with pl_legacy_patch():
            for key, value in torch.load('checkpoints/VTN_HCPF.ckpt',map_location='cpu')['state_dict'].items():
                new_state_dict[key.replace('model.','')] = value
        self.student.reset_head(226) # AUTSL
        self.student.load_state_dict(new_state_dict)
        self.student.classifier = nn.Identity()
        self.projection = nn.Linear(embed_size*2,embed_size*6)
        self.norm = MMTensorNorm(-1)
        self.relu = F.relu
    def forward(self,left = None,center = None,right = None,center_pf = None,left_pf = None,right_pf = None):  
        b, t, x, c, h, w = left.size()
        self.teacher.eval()
        teacher_features = None
        y = None
        teacher_features = self.teacher.forward_features(left = left,center = center,right = right,left_pf = left_pf,right_pf=right_pf,center_pf=center_pf)
        
        student_features = self.student(clip = center,poseflow = center_pf)['logits']
        student_features = self.projection(self.norm(student_features))
        if not self.training:
            y = self.teacher.classifier(student_features)
        
        return {
            'logits':y,
            'student_features': student_features,
            'teacher_features': teacher_features,
        }
    
class VTNHCPF_OneView_Sim_Knowledge_Distilation_Inference(nn.Module):
    def __init__(self, num_classes=199, num_heads=4, num_layers=2, embed_size=512, sequence_length=16, cnn='rn34',
                 freeze_layers=0, dropout=0, **kwargs):
        super().__init__()
        print("Model: VTNHCPF_OneView_Sim_Knowledge_Distilation_Inference")
        print("*"*20)
        self.student = VTNHCPF(num_classes,num_heads,num_layers,embed_size,sequence_length,cnn,freeze_layers,dropout,**kwargs)
        self.student.classifier = nn.Identity()
        self.projection = nn.Linear(embed_size*2,embed_size*6)
        self.norm = MMTensorNorm(-1)
        self.relu = F.relu
        self.classifier = LinearClassifier(embed_size*2*3, num_classes, dropout)
        print("*"*20)
    def forward(self,clip = None,poseflow = None):  
        center = clip
        center_pf = poseflow
        b, t, x, c, h, w = center.size()
       
        student_features = self.student(clip = center,poseflow = center_pf)['logits']
        student_features = self.projection(self.norm(student_features))
       
        y = self.classifier(student_features)
        
        return {
            'logits':y,
        }