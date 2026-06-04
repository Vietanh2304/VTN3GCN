import torch.nn as nn
import torch.optim as optim
from modelling.vtn_att_poseflow_model import (VTNHCPF,VTNHCPF_GCN,VTNHCPF_Three_View,VTN3GCN,
                                              VTNHCPF_OneView_Sim_Knowledge_Distilation,VTNHCPF_OneView_Sim_Knowledge_Distilation_Inference)
import torch
from trainer.tools import MyCustomLoss,OLM_Loss
from modelling.i3d import InceptionI3d,InceptionI3D_ThreeView,InceptionI3D_HandCrop,I3D_OneView_Sim_Knowledge_Distillation,I3D_OneView_Sim_Knowledge_Distillation_Inference,InceptionI3D_ThreeView_ShareWeights
from torchvision import models
from torch.nn import functional as F
from modelling.swin_transformer import SwinTransformer3d,SwinTransformer3d_ThreeView,SwinTransformer3d_HandCrop,VideoSwinTransformer_OneView_Sim_Knowledge_Distillation,VideoSwinTransformer_OneView_Sim_Knowledge_Distillation_Inference,SwinTransformer3d_ThreeView_ShareWeights
from collections import OrderedDict
from modelling.mvit_v2 import mvit_v2_s,MVitV2_ThreeView,MVitV2_HandCrop,MvitV2_OneView_Sim_Knowledge_Distillation,MvitV2_OneView_Sim_Knowledge_Distillation_Inference,MVitV2_ThreeView_ShareWeights
from pytorch_lightning.utilities.migration import pl_legacy_patch

def load_criterion(train_cfg):
    criterion = None
    if train_cfg['criterion'] == "MyCustomLoss":
        criterion = MyCustomLoss(label_smoothing=train_cfg['label_smoothing'])
    if train_cfg['criterion'] == "OLM_Loss": 
        criterion = OLM_Loss(label_smoothing=train_cfg['label_smoothing'])
    assert criterion is not None
    return criterion

def load_optimizer(train_cfg, model):
    # Revert ve paper goc: 1 group, alpha_t1 hoc cung lr voi backbone (1e-4)
    optimzer = None
    if train_cfg['optimzer'] == "SGD":
        optimzer = optim.SGD(model.parameters(), lr=train_cfg['learning_rate'], weight_decay=float(train_cfg['w_decay']), momentum=0.9, nesterov=True)
    if train_cfg['optimzer'] == "Adam":
        optimzer = optim.AdamW(model.parameters(), lr=train_cfg['learning_rate'], weight_decay=float(train_cfg['w_decay']))
    assert optimzer is not None
    return optimzer

def load_lr_scheduler(train_cfg,optimizer):
    scheduler = None
    if train_cfg['lr_scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=train_cfg['scheduler_factor'], patience=train_cfg['scheduler_patience'])
    if train_cfg['lr_scheduler'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_cfg['lr_step_size'], gamma=train_cfg['gamma'])
    assert scheduler is not None
    return scheduler

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Linear') != -1:
    try:
        if m.weight is not None:
            m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    except:
        pass


def load_model(cfg):
    if cfg['training']['pretrained']:
        print(f"load pretrained model: {cfg['training']['pretrained_model']}")
        if cfg['data']['model_name'] == 'vtn_att_poseflow':
            if ('.ckpt' in cfg['training']['pretrained_model']):
                # .ckpt format (Lightning) - VTN_HCPF AUTSL pretrained
                # Strip 'model.' prefix, skip classifier (226 vs 199), bottle_mm + position_encoding (potential shape diff)
                # Keep feature_extractor (ResNet ImageNet pretrained - quan trong nhat!)
                # Missing CBAM/ReZero/norm keys -> strict=False
                model = VTNHCPF(**cfg['model'], sequence_length=cfg['data']['num_output_frames'])
                new_state_dict = {}
                with pl_legacy_patch():
                    checkpoint = torch.load(cfg['training']['pretrained_model'], map_location='cpu')['state_dict']
                    for key, value in checkpoint.items():
                        new_key = key.replace('model.', '')
                        if new_key.startswith('classifier'):
                            continue
                        new_state_dict[new_key] = value
                missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
                model.reset_head(model.num_classes)
                print(f"[vtn_att_poseflow .ckpt] kept={len(new_state_dict)}, missing={len(missing)}, unexpected={len(unexpected)}")
                if len(missing) > 0:
                    print(f"  Missing (expected CBAM/ReZero/norm): {missing[:8]}{'...' if len(missing)>8 else ''}")
                if len(unexpected) > 0:
                    print(f"  Unexpected: {unexpected[:8]}{'...' if len(unexpected)>8 else ''}")
            else:
                model = VTNHCPF(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
                # Hot fix cause cannot find file .ckpt
                # Only need the line below in root repo:
                model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'))
                # Fix path:
                # new_state_dict = {}
                # for key, value in torch.load(cfg['training']['pretrained_model'],map_location='cpu').items():
                #         new_state_dict[key.replace('model.','')] = value
                # model.reset_head(226) # AUTSL
                # model.load_state_dict(new_state_dict)
                # model.reset_head(model.num_classes)
        elif cfg['data']['model_name'] == 'VTNGCN':
            model = VTNHCPF_GCN(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
            if '.ckpt' in cfg['training']['pretrained_model']: #Temporary
                new_state_dict = {}
                with pl_legacy_patch():
                    checkpoint = torch.load(cfg['training']['pretrained_model'], map_location='cpu')['state_dict']
                    for key, value in checkpoint.items():
                        new_key = key.replace('model.', '')
                        if not new_key.startswith('bottle_mm') and not new_key.startswith('self_attention_decoder') and not new_key.startswith('classifier'):
                        # if not new_key.startswith('bottle_mm') and not new_key.startswith('self_attention_decoder'):
                        # if not new_key.startswith('bottle_mm') and not new_key.startswith('classifier'):
                        # if not new_key.startswith('classifier'):
                            new_state_dict[new_key] = value
                model.load_state_dict(new_state_dict, strict=False)
                model.reset_head(model.num_classes)
            else:
                # .pth format (raw torch save - OrderedDict). Filter: classifier (diff num_classes),
                # bottle_mm, position_encoding (diff seq_len). Missing CBAM/ReZero/norm -> strict=False
                checkpoint = torch.load(cfg['training']['pretrained_model'], map_location='cpu')
                new_state_dict = {}
                for key, value in checkpoint.items():
                    # Match paper origin .ckpt branch: skip bottle_mm + self_attention_decoder + classifier
                    # AUTSL pretrained head not aligned with VSL semantic -> train from scratch
                    if not key.startswith('bottle_mm') and \
                       not key.startswith('self_attention_decoder') and \
                       not key.startswith('classifier'):
                        new_state_dict[key] = value
                missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
                model.reset_head(model.num_classes)
                print(f"[VTNGCN .pth] kept={len(new_state_dict)}, missing={len(missing)}, unexpected={len(unexpected)}")
                if len(missing) > 0:
                    print(f"  Missing (expected CBAM/ReZero/norm): {missing[:8]}{'...' if len(missing)>8 else ''}")
                if len(unexpected) > 0:
                    print(f"  Unexpected: {unexpected[:8]}{'...' if len(unexpected)>8 else ''}")

        elif cfg['data']['model_name'] == 'VTNHCPF_Three_view':
            model = VTNHCPF_Three_View(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
            if '.ckpt' in cfg['training']['pretrained_model']:
                new_state_dict = {}
                with pl_legacy_patch():
                    for key, value in torch.load(cfg['training']['pretrained_model'],map_location='cpu')['state_dict'].items():
                        new_state_dict[key.replace('model.','')] = value
                model.center.reset_head(226) # AUTSL
                model.left.reset_head(226) # AUTSL
                model.right.reset_head(226) # AUTSL
                # load autsl ckpt
                model.center.load_state_dict(new_state_dict)
                model.right.load_state_dict(new_state_dict)
                model.left.load_state_dict(new_state_dict)
                # add backbone
                model.add_backbone()
                # remove center, left and right backbone
                model.remove_head_and_backbone()
                model.freeze(layers = 0)
                print("Load VTNHCPF Three View")
            elif "IMAGENET" == cfg['training']['pretrained_model']:
                model.add_backbone()
                model.remove_head_and_backbone()
                print("Load VTNHCPF Three View IMAGENET")
            else:
                model.add_backbone()
                model.remove_head_and_backbone()
                model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'))
        
            print("Load VTNHCPF Three View")

        elif cfg['data']['model_name'] == 'VTN3GCN':
            model = VTN3GCN(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
            if '.ckpt' in cfg['training']['pretrained_model']:
                new_state_dict = {}
                with pl_legacy_patch():
                    for key, value in torch.load(cfg['training']['pretrained_model'],map_location='cpu')['state_dict'].items():
                        new_state_dict[key.replace('model.','')] = value
                model.center.reset_head(226) # AUTSL
                model.left.reset_head(226) # AUTSL
                model.right.reset_head(226) # AUTSL
                # load autsl ckpt
                model.center.load_state_dict(new_state_dict, strict=False)
                model.right.load_state_dict(new_state_dict, strict=False)
                model.left.load_state_dict(new_state_dict, strict=False)
                # add backbone
                model.add_backbone()
                # remove center, left and right backbone
                model.remove_head_and_backbone()
                model.freeze(layers = 0)
                print("Load VTN3GCN")
            elif "IMAGENET" == cfg['training']['pretrained_model']:
                model.add_backbone()
                model.remove_head_and_backbone()
                print("Load VTN3GCN IMAGENET")
            else:
                # Load .pth from one-view AUTSL → broadcast vào cả 3 view
                state_dict = torch.load(cfg['training']['pretrained_model'], map_location='cpu')
                # Filter classifier (size mismatch: 226 vs 3215 + 1024 vs 3072)
                filtered_sd = {
                    k: v for k, v in state_dict.items()
                    if not k.startswith('classifier')
                    and not k.startswith('bottle_mm')
                    and not k.startswith('self_attention_decoder.position_encoding')
                }
                # Load vào TỪNG view bằng cách thêm prefix
                missing_c, unexpected_c = model.center.load_state_dict(filtered_sd, strict=False)
                missing_l, unexpected_l = model.left.load_state_dict(filtered_sd, strict=False)
                missing_r, unexpected_r = model.right.load_state_dict(filtered_sd, strict=False)
                print(f"Loaded {len(filtered_sd)} keys (filtered classifier) into each of 3 views")
                print(f"Center: missing={len(missing_c)}, unexpected={len(unexpected_c)}")
                print(f"Left:   missing={len(missing_l)}, unexpected={len(unexpected_l)} (extra AAGCN expected)")
                print(f"Right:  missing={len(missing_r)}, unexpected={len(unexpected_r)} (extra AAGCN expected)")
                # Sau khi load 3 view xong, mới promote backbone lên top-level
                model.add_backbone()
                model.remove_head_and_backbone()
                model.freeze(layers=0)
                print("Load VTN3GCN from .pth one-view AUTSL — broadcast to 3 views")
        
            print("Load VTN3GCN")

        elif cfg['data']['model_name'] == 'InceptionI3d':
            model = InceptionI3d(**cfg['model'])
            if "autsl" in cfg['training']['pretrained_model'].split("/")[-1]:
                model.replace_logits(226)
                model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location=torch.device('cpu')))
                model.replace_logits(model._num_classes)
                print("Finetune fron AUTSL checkpoint")
            else:
                model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location=torch.device('cpu')))
        elif cfg['data']['model_name'] == 'InceptionI3d_ThreeView':
            model = InceptionI3D_ThreeView(**cfg['model'])
            model.remove_head()
            model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location=torch.device('cpu')),strict = True)
            print("Load pretrained I3D Three View")
        elif cfg['data']['model_name'] == 'InceptionI3D_ThreeView_ShareWeights':
            model = InceptionI3D_ThreeView_ShareWeights(**cfg['model'])
            model.remove_head()
            model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location=torch.device('cpu')),strict = True)
            print("Load InceptionI3D_ThreeView_ShareWeights")
        elif cfg['data']['model_name'] == 'InceptionI3D_HandCrop':
            model = InceptionI3D_HandCrop(**cfg['model'])
            if "autsl" in cfg['training']['pretrained_model'].split("/")[-1]:
                model.right.replace_logits(226)
                model.left.replace_logits(226)
                model.right.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location=torch.device('cpu')),strict = True)
                model.left.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location=torch.device('cpu')),strict = True)
                model.remove_head()
                model.freeze_and_remove(enpoints=0)
                print("Load I3D Hand Crop Pretrained on AUTSL")
            else:
                model.remove_head()
                model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location=torch.device('cpu')),strict = True)
                print("Load I3D Hand Crop")
                
        elif cfg['data']['model_name'] == 'swin_transformer':
            model = SwinTransformer3d(**cfg['model'])
            state_dict = torch.load(cfg['training']['pretrained_model'],map_location=torch.device('cpu'))
            if "autsl" in cfg['training']['pretrained_model'].split("/")[-1]:
                model.reset_head(226)
                model.load_state_dict(state_dict)
                model.reset_head(model.num_classes)
            else:
                model.load_state_dict(state_dict)
        elif cfg['data']['model_name'] == 'swin_transformer_3d_ThreeView':
            model = SwinTransformer3d_ThreeView(**cfg['model'])
            model.remove_head()
            model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'))
            print("Load Video Swin Transformer for Three view")
        elif cfg['data']['model_name'] == 'SwinTransformer3d_ThreeView_ShareWeights':
            model = SwinTransformer3d_ThreeView_ShareWeights(**cfg['model'])
            model.remove_head()
            model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'))
            print("Load SwinTransformer3d_ThreeView_ShareWeights model")
        elif cfg['data']['model_name'] == 'SwinTransformer3d_HandCrop':
            model = SwinTransformer3d_HandCrop(**cfg['model'])
            state_dict = torch.load(cfg['training']['pretrained_model'],map_location=torch.device('cpu'))
            if "autsl" in cfg['training']['pretrained_model'].split("/")[-1]:
                model.right.reset_head(226)
                model.left.reset_head(226)
                model.right.load_state_dict(state_dict,strict = True)
                model.left.load_state_dict(state_dict,strict = True)
                model.remove_head()
                model.freeze_and_remove(layers=4)
            else:
                model.remove_head()
                model.load_state_dict(state_dict,strict = True)

        elif cfg['data']['model_name'] == 'mvit_v2':
            model = mvit_v2_s(**cfg['model'])
            state_dict = torch.load(cfg['training']['pretrained_model'],map_location=torch.device('cpu'))
            if "autsl" in cfg['training']['pretrained_model'].split("/")[-1]:
                model.reset_head(226)
                model.load_state_dict(state_dict)
                model.reset_head(model.num_classes)
            else:
                model.load_state_dict(state_dict)
        elif cfg['data']['model_name'] == 'MVitV2_ThreeView':
            model = MVitV2_ThreeView(**cfg['model'])
            if "autsl" in cfg['training']['pretrained_model'].split("/")[-1]:
                state_dict = torch.load(cfg['training']['pretrained_model'],map_location='cpu')
                model.center.reset_head(226)      
                model.right.reset_head(226)      
                model.left.reset_head(226)            
                model.center.load_state_dict(state_dict,strict = True)
                model.right.load_state_dict(state_dict,strict = True)
                model.left.load_state_dict(state_dict,strict = True)
                model.remove_head()
                model.freeze_and_remove(layers=8)
                print("Load Mvit V2 Three View Pretrained on AUTSL")
            elif cfg['training']['pretrained_model'] == 'kinetics400':
                state_dict = models.video.MViT_V2_S_Weights.KINETICS400_V1.get_state_dict(progress=True)
                model.center.reset_head(400)      
                model.right.reset_head(400)      
                model.left.reset_head(400)            
                model.center.load_state_dict(state_dict,strict = True)
                model.right.load_state_dict(state_dict,strict = True)
                model.left.load_state_dict(state_dict,strict = True)
                model.remove_head()
                model.freeze_and_remove(layers=8)
                print("Load Mvit V2 Three View Pretrained on Kinetics400")
            else:
                model.remove_head()
                model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'))
                print("Load Mvit V2 Three View")
        elif cfg['data']['model_name'] == 'MVitV2_ThreeView_ShareWeights':
            model = MVitV2_ThreeView_ShareWeights(**cfg['model'])
            model.remove_head()
            model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'))
            print("Load Mvit V2 Three View Share Weights")
        elif cfg['data']['model_name'] == 'MVitV2_HandCrop':
            model = MVitV2_HandCrop(**cfg['model'])
            if "autsl" in cfg['training']['pretrained_model'].split("/")[-1]:
                pass
            else:
                model.remove_head()
                model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'))
    else:
        if cfg['data']['model_name'] == 'vtn_att_poseflow':
            model = VTNHCPF(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
        if cfg['data']['model_name'] == 'VTNHCPF_OneView_Sim_Knowledge_Distilation':
            model = VTNHCPF_OneView_Sim_Knowledge_Distilation(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
        if cfg['data']['model_name'] == 'VTNHCPF_OneView_Sim_Knowledge_Distilation_Inference':
            model = VTNHCPF_OneView_Sim_Knowledge_Distilation_Inference(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
            state_dict = torch.load("checkpoints/VTNHCPF_OneView_Sim_Knowledge_Distilation/VTNHCPF_OneView_Sim_Knowledge_Distilation with testing labels/best_checkpoints.pth",map_location='cpu')
            new_state_dict = {}
            for key,value in state_dict.items():
                if key.startswith('teacher'): # omit teacher state dict
                    if key.split('.')[1].startswith('classifier'): # save the classifier of the teacher model
                        new_state_dict[key.replace('teacher.','')] = value
                    continue
                new_state_dict[key] = value
            model.load_state_dict(new_state_dict)
        elif cfg['data']['model_name'] == 'VTNHCPF_Three_view':
            model = VTNHCPF_Three_View(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
            state_dict = torch.load("checkpoints/vtn_att_poseflow/vtn_att_poseflow autsl to vsl for one view/best_checkpoints.pth",map_location='cpu')
            model.center.load_state_dict(state_dict,strict = True)
            model.right.load_state_dict(state_dict,strict = True)
            model.left.load_state_dict(state_dict,strict = True)
            model.add_backbone()
            model.remove_head_and_backbone()
            model.freeze(layers = 0)
            print("Load VTNHCPF Three View")
        elif cfg['data']['model_name'] == 'VTN3GCN':
            if cfg['data']['center_kp']:
                model = VTN3GCN(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
                ckpt_path = "checkpoints/VTNGCN/VTNGCN finetune autsl to vsl for one view/best_checkpoints.pth"
                state_dict = torch.load(ckpt_path,map_location='cpu')
                print("Load VTN3GCN initialized weights: ",ckpt_path)
                model.center.load_state_dict(state_dict,strict = False)
                model.right.load_state_dict(state_dict,strict = False)
                model.left.load_state_dict(state_dict,strict = False)
                model.add_backbone()
                model.remove_head_and_backbone()
                model.freeze(layers = 0)
                print("Load VTN3GCN")
            else:
                model = VTN3GCN(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
                # vtngcn_ckpt_path = "checkpoints/VTNGCN/VTNGCN finetune autsl to vsl for one view/best_checkpoints.pth"
                vtngcn_ckpt_path = "checkpoints/VTNGCN/VTNGCN finetune autsl to vsl for one view no crop hand/best_checkpoints.pth"
                # vtngcn_ckpt_path = "checkpoints/VTNGCN/VTNGCN finetune autsl to vsl for one view no pose flow/best_checkpoints.pth"
                # vtngcn_ckpt_path = "checkpoints/VTNGCN/VTNGCN finetune autsl to vsl for one view no crop hand and pose flow/best_checkpoints.pth"
                vtn_ckpt_path = "checkpoints/vtn_att_poseflow/vtn_att_poseflow autsl to vsl for one view/best_checkpoints.pth"
                vtngcn_state_dict = torch.load(vtngcn_ckpt_path,map_location='cpu')
                vtn_state_dict = torch.load(vtn_ckpt_path,map_location='cpu')
                new_vtn_state_dict = {}
                for key, value in vtn_state_dict.items():
                    if not key.startswith('self_attention_decoder') and not key.startswith('bottle_mm') and not key.startswith('classifier'):
                        new_vtn_state_dict[key] = value
                print("Load VTN3GCN initialized weights: ",vtngcn_ckpt_path)
                print("Load VTN3GCN initialized weights: ",vtn_ckpt_path)
                model.center.load_state_dict(new_vtn_state_dict,strict = False)
                model.right.load_state_dict(vtngcn_state_dict,strict = False)
                model.left.load_state_dict(vtngcn_state_dict,strict = False)
                model.add_backbone()
                model.remove_head_and_backbone()
                model.freeze(layers = 0)
                print("Load VTN3GCN")

        elif cfg['data']['model_name'] == 'InceptionI3d':
            model = InceptionI3d(**cfg['model'])
            new_dict = {}
            for key,value in torch.load('pretrained_models/InceptionI3D/rgb_charades.pt',map_location=torch.device('cpu')).items():
                if key.startswith('logits'):
                    continue
                new_dict[key] = value
            model.load_state_dict(new_dict,strict = False)
        elif cfg['data']['model_name'] == 'InceptionI3d_ThreeView':
            model = InceptionI3D_ThreeView(**cfg['model'])
            state_dict = torch.load("checkpoints/InceptionI3d/I3D finetune from autsl for one view/best_checkpoints.pth",map_location='cpu')
            model.center.load_state_dict(state_dict,strict = True)
            model.right.load_state_dict(state_dict,strict = True)
            model.left.load_state_dict(state_dict,strict = True)
            model.remove_head()
            model.freeze_and_remove(enpoints=6)
            print("Load I3D Three View")
        elif cfg['data']['model_name'] == 'InceptionI3D_ThreeView_ShareWeights':
            model = InceptionI3D_ThreeView_ShareWeights(**cfg['model'])
            state_dict = torch.load("checkpoints/InceptionI3d/I3D finetune from autsl for one view/best_checkpoints.pth",map_location='cpu')
            model.encoder.load_state_dict(state_dict,strict = True)
            model.remove_head()
            model.freeze_and_remove(enpoints=6)
            print("Load InceptionI3D_ThreeView_ShareWeights")
        elif cfg['data']['model_name'] == 'InceptionI3D_HandCrop':
            model = InceptionI3D_HandCrop(**cfg['model'])
            new_dict = {}
            for key,value in torch.load('pretrained_models/InceptionI3D/rgb_charades.pt',map_location=torch.device('cpu')).items():
                if key.startswith('logits'):
                    continue
                new_dict[key] = value
            model.remove_head()
            model.right.load_state_dict(new_dict,strict = True)
            model.left.load_state_dict(new_dict,strict = True)
            model.freeze_and_remove(enpoints=0)
            print("Load I3D Hand Crop")
        elif cfg['data']['model_name'] == 'I3D_OneView_Sim_Knowledge_Distillation':
            model = I3D_OneView_Sim_Knowledge_Distillation(**cfg['model'])
        elif cfg['data']['model_name'] == 'I3D_OneView_Sim_Knowledge_Distillation_Inference':
            model = I3D_OneView_Sim_Knowledge_Distillation_Inference(**cfg['model'])
            state_dict = torch.load("checkpoints/I3D_OneView_Sim_Knowledge_Distillation/I3D_OneView_Sim_Knowledge_Distillation_v1/best_checkpoints.pth",map_location='cpu')
            new_state_dict = {}
            for key,value in state_dict.items():
                if key.startswith('teacher'): # omit teacher state dict
                    if key.split('.')[1].startswith('classififer'): # save the classifier of the teacher model
                        new_state_dict[key.replace('teacher.','')] = value
                    continue
                new_state_dict[key] = value
            model.load_state_dict(new_state_dict)
        
        elif cfg['data']['model_name'] == 'swin_transformer':
            model = SwinTransformer3d(**cfg['model'])
            weights = models.video.Swin3D_T_Weights.DEFAULT.get_state_dict(progress=True)
            model.reset_head(400)
            model.load_state_dict(weights)
            model.reset_head(model.num_classes)
        elif cfg['data']['model_name'] == 'swin_transformer_3d_ThreeView':
            model = SwinTransformer3d_ThreeView(**cfg['model'])
            state_dict = torch.load("checkpoints/swin_transformer/Swin Transformer 3D Tiny for one view finetune from autsl /best_checkpoints.pth",map_location='cpu')
            model.center.load_state_dict(state_dict,strict = True)
            model.right.load_state_dict(state_dict,strict = True)
            model.left.load_state_dict(state_dict,strict = True)
            model.remove_head()
            model.freeze_and_remove(layers=4)
        elif cfg['data']['model_name'] == 'SwinTransformer3d_ThreeView_ShareWeights':
            model = SwinTransformer3d_ThreeView_ShareWeights(**cfg['model'])
            state_dict = torch.load("checkpoints/swin_transformer/Swin Transformer 3D Tiny for one view finetune from autsl /best_checkpoints.pth",map_location='cpu')
            model.encoder.load_state_dict(state_dict,strict = True)
            model.remove_head()
            model.freeze_and_remove(layers=4)
            print("Load SwinTransformer3d_ThreeView_ShareWeights model")
        
        elif cfg['data']['model_name'] == 'SwinTransformer3d_HandCrop':
            model = SwinTransformer3d_HandCrop(**cfg['model'])
            state_dict  = models.video.Swin3D_T_Weights.DEFAULT.get_state_dict(progress=True)
            model.right.reset_head(400)
            model.left.reset_head(400)
            model.right.load_state_dict(state_dict,strict = True)
            model.left.load_state_dict(state_dict,strict = True)
            model.remove_head()
            model.freeze_and_remove(layers=4)
        elif cfg['data']['model_name'] == 'VideoSwinTransformer_OneView_Sim_Knowledge_Distillation':
            model = VideoSwinTransformer_OneView_Sim_Knowledge_Distillation(**cfg['model'])
        elif cfg['data']['model_name'] == 'VideoSwinTransformer_OneView_Sim_Knowledge_Distillation_Inference':
            model = VideoSwinTransformer_OneView_Sim_Knowledge_Distillation_Inference(**cfg['model'])
            state_dict = torch.load("checkpoints/VideoSwinTransformer_OneView_Sim_Knowledge_Distillation/VideoSwinTransformer_OneView_Sim_Knowledge_Distillation/best_checkpoints.pth",map_location='cpu')
            new_state_dict = {}
            for key,value in state_dict.items():
                if key.startswith('teacher'): # omit teacher state dict
                    if key.split('.')[1].startswith('classififer'): # save the classifier of the teacher model
                        new_state_dict[key.replace('teacher.','')] = value
                    continue
                new_state_dict[key] = value
            model.load_state_dict(new_state_dict)
        elif cfg['data']['model_name'] == 'mvit_v2':
            model = mvit_v2_s(**cfg['model'])
            model.reset_head(400)
            weights = models.video.MViT_V2_S_Weights.KINETICS400_V1.get_state_dict(progress=True)
            model.load_state_dict(weights)
            model.reset_head(model.num_classes)
        elif cfg['data']['model_name'] == 'MVitV2_ThreeView':
            model = MVitV2_ThreeView(**cfg['model'])
            state_dict = torch.load("checkpoints/mvit_v2/MVIT V2 Small for one view finetune from AUTSL/best_checkpoints.pth",map_location='cpu')
            model.center.load_state_dict(state_dict,strict = True)
            model.right.load_state_dict(state_dict,strict = True)
            model.left.load_state_dict(state_dict,strict = True)
            model.remove_head()
            model.freeze_and_remove(layers=8)
            model.count()
        elif cfg['data']['model_name'] == 'MVitV2_ThreeView_ShareWeights':
            model = MVitV2_ThreeView_ShareWeights(**cfg['model'])
            state_dict = torch.load("checkpoints/mvit_v2/MVIT V2 Small for one view finetune from AUTSL/best_checkpoints.pth",map_location='cpu')
            model.encoder.load_state_dict(state_dict,strict = True)
            model.remove_head()
            model.freeze_layers(8)
            model.count()
       
        elif cfg['data']['model_name'] == 'MVitV2_HandCrop':
            model = MVitV2_HandCrop(**cfg['model'])
            state_dict = models.video.MViT_V2_S_Weights.KINETICS400_V1.get_state_dict(progress=True)
            model.left.reset_head(400)
            model.right.reset_head(400)
            model.right.load_state_dict(state_dict,strict = True)
            model.left.load_state_dict(state_dict,strict = True)
            model.remove_head()
            model.freeze_and_remove(layers=8)
        elif cfg['data']['model_name'] == 'MvitV2_OneView_Sim_Knowledge_Distillation':
            model = MvitV2_OneView_Sim_Knowledge_Distillation(**cfg['model'])
        elif cfg['data']['model_name'] == 'MvitV2_OneView_Sim_Knowledge_Distillation_Inference':
            model = MvitV2_OneView_Sim_Knowledge_Distillation_Inference(**cfg['model'])
            state_dict = torch.load("checkpoints/MvitV2_OneView_Sim_Knowledge_Distillation/MvitV2_OneView_Sim_Knowledge_Distillation/best_checkpoints.pth",map_location='cpu')
            new_state_dict = {}
            for key,value in state_dict.items():
                if key.startswith('teacher'): # omit teacher state dict
                    if key.split('.')[1].startswith('classififer'): # save the classifier of the teacher model
                        new_state_dict[key.replace('teacher.','')] = value
                    continue
                new_state_dict[key] = value
            model.load_state_dict(new_state_dict)
        
        
  




    assert model is not None
    print("loaded model")
    return model
