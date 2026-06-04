from dataset.videoLoader import load_batch_video,get_selected_indexs,pad_array,pad_index
from dataset.dataset import build_dataset
import torch
from functools import partial
import random
import numpy as np
import os
import torchvision
import json
from utils.video_augmentation import DeleteFlowKeypoints,ToFloatTensor,Compose
import math
from transformers import AutoTokenizer 
from PIL import Image        
import random
import cv2



def vtn_pf_collate_fn_(batch):
    labels = torch.stack([s[2] for s in batch],dim = 0)
    clip = torch.stack([s[0] for s in batch],dim = 0)
    poseflow = torch.stack([s[1] for s in batch],dim = 0) 
    return {'clip':clip,'poseflow':poseflow},labels

def vtn_gcn_collate_fn_(batch):
    clip = torch.stack([s[0] for s in batch], dim = 0)
    poseflow = torch.stack([s[1] for s in batch], dim = 0)
    keypoints = torch.stack([s[2] for s in batch], dim = 0)
    labels = torch.stack([s[3] for s in batch], dim = 0)
    return {'clip':clip, 'poseflow':poseflow, 'keypoints':keypoints},labels

def vtn_rgb_heat_collate_fn_(batch):
    heatmap = torch.stack([s[0] for s in batch], dim=0)
    rgb = torch.stack([s[1] for s in batch],dim = 0)
    pf = torch.stack([s[2] for s in batch],dim = 0)
    labels = torch.stack([s[3] for s in batch], dim=0)
    return {'heatmap':heatmap,'rgb':rgb,'pf':pf},labels

def gcn_bert_collate_fn_(batch):
    labels = torch.stack([s[1] for s in batch],dim = 0)
    keypoints = torch.stack([s[0] for s in batch],dim = 0) # bs t n c
   
    return {'keypoints':keypoints},labels

def three_viewpoints_collate_fn_(batch):
    center_video = torch.stack([s[0] for s in batch],dim = 0)
    left_video = torch.stack([s[1] for s in batch],dim = 0)
    right_video = torch.stack([s[2] for s in batch],dim = 0)
    labels = torch.stack([s[3] for s in batch],dim = 0)
    
    return {'left':left_video,'center':center_video,'right':right_video},labels

def i3d_collate_fn_(batch):
    clip = torch.stack([s[0] for s in batch],dim = 0)
    labels = torch.stack([s[1] for s in batch],dim = 0)
    
    return {'clip':clip},labels

def videomae_collate_fn_(batch):
    clip = torch.stack([s[0] for s in batch],dim = 0)
    mask = torch.stack([s[1] for s in batch],dim = 0)
    labels = torch.stack([s[2] for s in batch],dim = 0)
    return {'clip':clip,'mask':mask},labels

def swin_transformer_collate_fn_(batch):
    clip = torch.stack([s[0] for s in batch],dim = 0).permute(0,2,1,3,4) # b,t,c,h,w -> b,c,t,h,w
    labels = torch.stack([s[1] for s in batch],dim = 0)
    return {'clip':clip},labels

def mvit_transformer_collate_fn_(batch):
    clip = torch.stack([s[0] for s in batch],dim = 0).permute(0,2,1,3,4) # b,t,c,h,w -> b,c,t,h,w
    labels = torch.stack([s[1] for s in batch],dim = 0)
    return {'clip':clip},labels

def vtn_hc_pf_three_view_collate_fn_(batch):
    center_video = torch.stack([s[0] for s in batch],dim = 0)
    left_video = torch.stack([s[2] for s in batch],dim = 0)
    right_video = torch.stack([s[4] for s in batch],dim = 0)
    labels = torch.stack([s[6] for s in batch],dim = 0)

    center_pf = torch.stack([s[1] for s in batch],dim = 0)
    left_pf = torch.stack([s[3] for s in batch],dim = 0)
    right_pf = torch.stack([s[5] for s in batch],dim = 0)
    
    return {'left':left_video,'center':center_video,'right':right_video,'center_pf':center_pf,'left_pf':left_pf,'right_pf':right_pf},labels

def vtn_3_gcn_collate_fn_(batch):
    center_video = torch.stack([s[0] for s in batch],dim = 0)
    left_video = torch.stack([s[3] for s in batch],dim = 0)
    right_video = torch.stack([s[6] for s in batch],dim = 0)
    
    center_pf = torch.stack([s[1] for s in batch],dim = 0)
    left_pf = torch.stack([s[4] for s in batch],dim = 0)
    right_pf = torch.stack([s[7] for s in batch],dim = 0)

    center_kp = torch.stack([s[2] for s in batch],dim = 0)
    left_kp = torch.stack([s[5] for s in batch],dim = 0)
    right_kp = torch.stack([s[8] for s in batch],dim = 0)

    labels = torch.stack([s[9] for s in batch],dim = 0)

    return {'left':left_video,'center':center_video,'right':right_video,'center_pf':center_pf,'left_pf':left_pf,'right_pf':right_pf,
            'center_kp':center_kp,'left_kp':left_kp,'right_kp':right_kp},labels

def distilation_collate_fn_(batch):
    center_video = torch.stack([s[0] for s in batch],dim = 0)
    left_video = torch.stack([s[2] for s in batch],dim = 0)
    right_video = torch.stack([s[4] for s in batch],dim = 0)

    center_pf = torch.stack([s[1] for s in batch],dim = 0)
    left_pf = torch.stack([s[3] for s in batch],dim = 0)
    right_pf = torch.stack([s[5] for s in batch],dim = 0)

    center_clip_no_crop_hand = torch.stack([s[6] for s in batch],dim = 0)
    labels = torch.stack([s[7] for s in batch],dim = 0)
    
    return {'left':left_video,'center':center_video,'right':right_video,
            'center_pf':center_pf,'left_pf':left_pf,'right_pf':right_pf,
            'center_clip_no_crop_hand':center_clip_no_crop_hand
            },labels

def build_dataloader(cfg, split, is_train=True, model = None,labels = None):
    dataset = build_dataset(cfg['data'], split,model,train_labels = labels)

    if cfg['data']['model_name'] == 'VTN_RGBheat' or cfg['data']['model_name'] == '2s-CrossVTN':
        collate_func = vtn_rgb_heat_collate_fn_
    if cfg['data']['model_name'] == 'VTNGCN' or cfg['data']['model_name'] == 'VTNGCN_Combine':
        collate_func = vtn_gcn_collate_fn_
    if cfg['data']['model_name'] == 'VTN3GCN' or cfg['data']['model_name'] == 'VTN3GCN_v2':
        collate_func = vtn_3_gcn_collate_fn_
    if cfg['data']['model_name'] == 'vtn_att_poseflow' or 'HandCrop' in cfg['data']['model_name'] or cfg['data']['model_name'] == 'VTNHCPF_OneView_Sim_Knowledge_Distilation_Inference':
        collate_func = vtn_pf_collate_fn_
    if cfg['data']['model_name'] == 'gcn_bert':
        collate_func = gcn_bert_collate_fn_
    
    distillation_models = ['MvitV2_OneView_Sim_Knowledge_Distillation','I3D_OneView_Sim_Knowledge_Distillation','VideoSwinTransformer_OneView_Sim_Knowledge_Distillation']

    if 'ThreeView' in cfg['data']['model_name'] or cfg['data']['model_name'] in distillation_models:
        collate_func = three_viewpoints_collate_fn_
    if cfg['data']['model_name'] == 'InceptionI3d' or cfg['data']['model_name'] == 'I3D_OneView_Sim_Knowledge_Distillation_Inference':
        collate_func = i3d_collate_fn_
    if cfg['data']['model_name'] == 'videomae':
        collate_func = videomae_collate_fn_
    if cfg['data']['model_name'] == 'swin_transformer' or cfg['data']['model_name'] == 'VideoSwinTransformer_OneView_Sim_Knowledge_Distillation_Inference':
        collate_func = swin_transformer_collate_fn_
    if 'mvit' in cfg['data']['model_name'] or cfg['data']['model_name'] == 'MvitV2_OneView_Sim_Knowledge_Distillation_Inference':
        collate_func = mvit_transformer_collate_fn_
    if cfg['data']['model_name']  == 'VTNHCPF_Three_view' or cfg['data']['model_name'] == 'VTNHCPF_OneView_Sim_Knowledge_Distilation':
        collate_func = vtn_hc_pf_three_view_collate_fn_

    dataloader = torch.utils.data.DataLoader(dataset,
                                            collate_fn = collate_func,
                                            batch_size = cfg['training']['batch_size'],
                                            num_workers = cfg['training'].get('num_workers',2),
                                            shuffle = is_train,
                                            prefetch_factor = cfg['training'].get('prefetch_factor',2),
                                            pin_memory=True,
                                            persistent_workers =  2,
                                            # sampler = sampler
                                            )

    return dataloader
