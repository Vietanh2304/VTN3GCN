import os
import h5py
import numpy as np
import torch
from torch.utils.data import  Dataset
import pandas as pd
from transformers import AutoTokenizer
from dataset.videoLoader import get_selected_indexs,pad_index
import cv2
import torchvision
from dataset.utils import crop_hand
import json
from PIL import Image
from utils.video_augmentation import DeleteFlowKeypoints,ToFloatTensor,Compose
import glob
import time
from decord import VideoReader
import threading
import math
from utils.video_augmentation import *
from utils.video_augmentation import (
    TemporalSpeedJitter, KeypointGaussianNoise, KeypointDropJoint,
    HandCropRandomErasing, TemporalFrameDropout, SyncedMultiViewTransform
)
class VTNHCPF_ThreeViewsData(Dataset):
    def __init__(self, base_url,split,dataset_cfg,**kwargs):
        
        if dataset_cfg is None:
            self.train_labels = pd.read_csv(os.path.join(base_url,f'{split}.csv'),sep=',')
        else:
            if dataset_cfg['dataset_name'] == "VN_SIGN":
                print("Label: ",os.path.join(base_url,f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"))
                self.train_labels = pd.read_csv(os.path.join(base_url,f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"),sep=',')
                # if split == 'test':
                #      self.train_labels = pd.concat([self.train_labels] * 5, ignore_index=True)
            elif dataset_cfg['dataset_name'] == "AUTSL":
                self.train_labels = pd.read_csv(os.path.join(base_url,f'{split}.csv'),sep=',')
                self.labels = pd.read_csv(os.path.join(base_url,f'SignList_ClassId_TR_EN.csv'),sep=',')
       
        print(split,len(self.train_labels))
        self.split = split
        if split == 'train':
            self.is_train = True
        else:
            self.is_train = False
        self.base_url = base_url
        self.data_cfg = dataset_cfg
        self.data_name = dataset_cfg['dataset_name']
        self.pose_transform  = Compose(
            DeleteFlowKeypoints(list(range(11, 91))),  # delete foot+face → keep body[0:11] + hands[91:133] = 53 kp
            ToFloatTensor()
        )
        # HDF5 paths — opened lazily per worker in __getitem__
        self.wholebody_h5_path = os.path.join(base_url, dataset_cfg['wholebody_h5_dir'].split('MM-WLAuslan/')[-1], f'{split}.h5') if 'wholebody_h5_dir' in dataset_cfg else None
        self.poseflow_h5_path = os.path.join(base_url, dataset_cfg['poseflow_h5_dir'].split('MM-WLAuslan/')[-1], f'{split}.h5') if 'poseflow_h5_dir' in dataset_cfg else None
        self._wholebody_h5 = None
        self._poseflow_h5 = None
        self.transform = self.build_transform(split)
    def build_transform(self,split):
        if split == 'train':
            print("Build train transform")
            transform = Compose(
                                Scale(self.data_cfg['vid_transform']['IMAGE_SIZE'] * 8 // 7),
                                MultiScaleCrop((self.data_cfg['vid_transform']['IMAGE_SIZE'], self.data_cfg['vid_transform']['IMAGE_SIZE']), scales),
                                # RandomHorizontalFlip(),  # disabled: SLR has dominant hand 
                                RandomRotate(p=0.3),
                                RandomShear(0.3,0.3,p = 0.3),
                                Salt( p = 0.3),
                                GaussianBlur( sigma=1,p = 0.3),
                                ColorJitter(0.5, 0.5, 0.5,p = 0.3),
                                ToFloatTensor(), PermuteImage(),
                                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'],self.data_cfg['vid_transform']['NORM_STD_IMGNET']))
        else:
            print("Build test/val transform")
            transform = Compose(
                                Scale(self.data_cfg['vid_transform']['IMAGE_SIZE'] * 8 // 7), 
                                CenterCrop(self.data_cfg['vid_transform']['IMAGE_SIZE']), 
                                ToFloatTensor(),
                                PermuteImage(),
                                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'],self.data_cfg['vid_transform']['NORM_STD_IMGNET']))
        return transform
    
    def _ensure_h5_open(self):
        if self._wholebody_h5 is None and self.wholebody_h5_path:
            self._wholebody_h5 = h5py.File(self.wholebody_h5_path, 'r')
        if self._poseflow_h5 is None and self.poseflow_h5_path:
            self._poseflow_h5 = h5py.File(self.poseflow_h5_path, 'r')

    def count_frames(self,video_path):
        cap = cv2.VideoCapture(video_path)
        # Đọc kích thước của video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # HDF5 frame count lookup (replaces glob on poses/ folder)
        video_id = os.path.basename(video_path).replace('.mp4', '')
        self._ensure_h5_open()
        if self._wholebody_h5 is not None and video_id in self._wholebody_h5:
            n_poses = self._wholebody_h5[video_id]['wholebody_threshold_02'].shape[0]
        else:
            n_poses = 0
        total_frames = min(total_frames, n_poses)
        return total_frames,width,height
    def read_one_view(self,name,selected_index,width,height):
       
        clip = []
        poseflow_clip = []
        missing_wrists_left = []
        missing_wrists_right = []
       
        if self.data_cfg['dataset_name'] == "VN_SIGN":
            path = f'{self.base_url}/videos/{name}'   
        vr = VideoReader(path,width=320, height=256)
        frames = vr.get_batch(selected_index).asnumpy()
        self._ensure_h5_open()
        for frame,frame_index in zip(frames,selected_index):
            if self.data_cfg['crop_two_hand']:
                video_id = name.replace(".mp4", "")
                keypoints_full = self._wholebody_h5[video_id]['wholebody_threshold_02'][frame_index]  # (133, 3)
                x = 320 * keypoints_full[:, 0] / width
                y = 256 * keypoints_full[:, 1] / height
                keypoints = np.stack((x, y), axis=0)  # (2, 133)

            crops = None
            if self.data_cfg['crop_two_hand']:
                crops,missing_wrists_left,missing_wrists_right = crop_hand(frame,keypoints,self.data_cfg['WRIST_DELTA'],self.data_cfg['SHOULDER_DIST_EPSILON'],
                                                                       self.transform,len(clip),missing_wrists_left,missing_wrists_right)
            else:
                crops = self.transform(frame)
            clip.append(crops)

            # Let's say the first frame has a pose flow of 0
            frame_index_poseflow = frame_index
            video_id = name.replace(".mp4", "")
            poseflow_data = self._poseflow_h5[video_id]['poseflow']  # (T-1, 133, 2)
            if frame_index_poseflow > 0 and (frame_index_poseflow - 1) < poseflow_data.shape[0]:
                poseflow = poseflow_data[frame_index_poseflow - 1].copy()  # (133, 2)
                poseflow[:, 0] /= math.pi
            else:
                poseflow = np.zeros((133, 2))

            pose_transform = Compose(
                DeleteFlowKeypoints(list(range(11, 91))),  # delete foot+face → keep body[0:11] + hands[91:133] = 53 kp
                ToFloatTensor()
            )
            poseflow = pose_transform(poseflow).view(-1)
            poseflow_clip.append(poseflow)
            
        clip = torch.stack(clip,dim = 0)
        poseflow = torch.stack(poseflow_clip, dim=0)
        return clip,poseflow

    def read_videos(self,center,left,right):
        index_setting = self.data_cfg['transform_cfg'].get('index_setting', ['consecutive','pad','central','pad'])
        # 
        vlen1,c_width,c_height = self.count_frames(os.path.join(self.base_url,'videos',center))
        vlen2,l_width,l_height = self.count_frames(os.path.join(self.base_url,'videos',left))
        vlen3,r_width,r_height = self.count_frames(os.path.join(self.base_url,'videos',right))

       
        min_vlen = min(vlen1,min(vlen2,vlen3))
        max_vlen = max(vlen1,max(vlen2,vlen3))
        if max_vlen - min_vlen < 10:
            selected_index, pad = get_selected_indexs(min_vlen - 3,self.data_cfg['num_output_frames'],self.is_train,index_setting,temporal_stride=self.data_cfg['temporal_stride'])
            
            if pad is not None:
                print("Vlen",center,left,right)
                selected_index  = pad_index(selected_index,pad).tolist()
        
            center_video,center_pf = self.read_one_view(center,selected_index,width=c_width,height=c_height)
            
            left_video,left_pf = self.read_one_view(left,selected_index,width=l_width,height=l_height)

            right_video,right_pf = self.read_one_view(right,selected_index,width=r_width,height=r_height)
        else:
            selected_index, pad = get_selected_indexs(vlen1 - 3,self.data_cfg['num_output_frames'],self.is_train,index_setting,temporal_stride=self.data_cfg['temporal_stride'])
            
            if pad is not None:
                print("Vlen",center)
                selected_index  = pad_index(selected_index,pad).tolist()

            center_video,center_pf = self.read_one_view(center,selected_index,width=c_width,height=c_height)

            selected_index, pad = get_selected_indexs(vlen2 - 3,self.data_cfg['num_output_frames'],self.is_train,index_setting,temporal_stride=self.data_cfg['temporal_stride'])
            
            if pad is not None:
                print("Vlen",left)
                selected_index  = pad_index(selected_index,pad).tolist()
            
            
            left_video,left_pf = self.read_one_view(left,selected_index,width=l_width,height=l_height)

            selected_index, pad = get_selected_indexs(vlen3-3,self.data_cfg['num_output_frames'],self.is_train,index_setting,temporal_stride=self.data_cfg['temporal_stride'])
            
            if pad is not None:
                print("Vlen",right)
                selected_index  = pad_index(selected_index,pad).tolist()

            right_video,right_pf = self.read_one_view(right,selected_index,width=r_width,height=r_height)

       

        return center_video,center_pf,left_video,left_pf,right_video,right_pf
        # return 1


    def __getitem__(self, idx):
        self.transform.randomize_parameters()
       
        center,left,right,label = self.train_labels.iloc[idx].values
       
        center_video,center_pf,left_video,left_pf,right_video,right_pf = self.read_videos(center,left,right)
        
        return center_video,center_pf,left_video,left_pf,right_video,right_pf,torch.tensor(label)
     
    
    def __len__(self):
        return len(self.train_labels)
    
class VTN3GCNData(Dataset):
    def __init__(self, base_url,split,dataset_cfg,**kwargs):
        
        if dataset_cfg is None:
            self.train_labels = pd.read_csv(os.path.join(base_url,f'{split}.csv'),sep=',')
        else:
            if dataset_cfg['dataset_name'] == "VN_SIGN":
                print("Label: ",os.path.join(base_url,f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"))
                self.train_labels = pd.read_csv(os.path.join(base_url,f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"),sep=',')
                # if split == 'test':
                #      self.train_labels = pd.concat([self.train_labels] * 5, ignore_index=True)
            elif dataset_cfg['dataset_name'] == "AUTSL":
                self.train_labels = pd.read_csv(os.path.join(base_url,f'{split}.csv'),sep=',')
                self.labels = pd.read_csv(os.path.join(base_url,f'SignList_ClassId_TR_EN.csv'),sep=',')
       
        print(split,len(self.train_labels))
        self.split = split
        if split == 'train':
            self.is_train = True
        else:
            self.is_train = False
        self.base_url = base_url
        self.data_cfg = dataset_cfg
        self.data_name = dataset_cfg['dataset_name']
        self.pose_transform  = Compose(
            DeleteFlowKeypoints(list(range(11, 91))),  # delete foot+face → keep body[0:11] + hands[91:133] = 53 kp
            ToFloatTensor()
        )
        # HDF5 paths — opened lazily per worker in __getitem__
        self.wholebody_h5_path = os.path.join(dataset_cfg['wholebody_h5_dir'], f'{split}.h5')
        self.poseflow_h5_path = os.path.join(dataset_cfg['poseflow_h5_dir'], f'{split}.h5')
        self.hand_kp_h5_path = os.path.join(dataset_cfg['hand_kp_h5_dir'], f'{split}.h5')
        self._wholebody_h5 = None
        self._poseflow_h5 = None
        self._hand_kp_h5 = None
        self.transform = self.build_transform(split)
        
        if self.is_train:
            self.global_augs = [TemporalSpeedJitter(p=0.8), TemporalFrameDropout(p=0.3)]
            self.per_view_augs = [
                KeypointGaussianNoise(p=0.5),
                KeypointDropJoint(p=0.4),
                HandCropRandomErasing(p=0.5),
            ]
            self.synced_spatial = SyncedMultiViewTransform(transforms_list=[]) # Tránh crash do spatial đã chạy ở build_transform
        else:
            self.global_augs = []
            self.per_view_augs = []
            self.synced_spatial = None

    def build_transform(self,split):
        if split == 'train':
            print("Build train transform")
            transform = Compose(
                                Scale(self.data_cfg['vid_transform']['IMAGE_SIZE'] * 8 // 7),
                                MultiScaleCrop((self.data_cfg['vid_transform']['IMAGE_SIZE'], self.data_cfg['vid_transform']['IMAGE_SIZE']), scales),
                                ColorJitter(0.5, 0.5, 0.5, p = 0.3),
                                ToFloatTensor(), PermuteImage(),
                                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'],self.data_cfg['vid_transform']['NORM_STD_IMGNET']))
        else:
            print("Build test/val transform")
            transform = Compose(
                                Scale(self.data_cfg['vid_transform']['IMAGE_SIZE'] * 8 // 7), 
                                CenterCrop(self.data_cfg['vid_transform']['IMAGE_SIZE']), 
                                ToFloatTensor(),
                                PermuteImage(),
                                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'],self.data_cfg['vid_transform']['NORM_STD_IMGNET']))
        return transform
    
    def _ensure_h5_open(self):
        if self._wholebody_h5 is None:
            self._wholebody_h5 = h5py.File(self.wholebody_h5_path, 'r')
        if self._poseflow_h5 is None:
            self._poseflow_h5 = h5py.File(self.poseflow_h5_path, 'r')
        if self._hand_kp_h5 is None:
            self._hand_kp_h5 = h5py.File(self.hand_kp_h5_path, 'r')

    def count_frames(self,video_path):
        cap = cv2.VideoCapture(video_path)
        # Đọc kích thước của video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # HDF5 frame count lookup (replaces glob on poses/ folder)
        video_id = os.path.basename(video_path).replace('.mp4', '')
        self._ensure_h5_open()
        if video_id in self._wholebody_h5:
            n_poses = self._wholebody_h5[video_id]['wholebody_threshold_02'].shape[0]
        else:
            n_poses = 0
        total_frames = min(total_frames, n_poses)
        return total_frames,width,height
    def transform_handflow(self, handflow):
        # handflow: numpy (46, 2) raw pixel coords from HDF5
        handflow_tensor = torch.tensor(handflow, dtype=torch.float32).transpose(0, 1)  # (2, 46)
        # Normalize to [0, ~1] — match AAGCN AUTSL training distribution
        handflow_tensor = handflow_tensor / 256.0
        return handflow_tensor
    def read_one_view(self,name,selected_index,width,height):
       
        clip = []
        poseflow_clip = []
        handkp_clip = []
        missing_wrists_left = []
        missing_wrists_right = []
       
        if self.data_cfg['dataset_name'] == "VN_SIGN":
            path = f'{self.base_url}/videos/{name}'   
        vr = VideoReader(path,width=320, height=256)
        frames = vr.get_batch(selected_index).asnumpy()
        self._ensure_h5_open()
        for frame,frame_index in zip(frames,selected_index):
            if self.data_cfg['crop_two_hand']:
                video_id = name.replace(".mp4", "")
                keypoints_full = self._wholebody_h5[video_id]['wholebody_threshold_02'][frame_index]  # (133, 3)
                x = 320 * keypoints_full[:, 0] / width
                y = 256 * keypoints_full[:, 1] / height
                keypoints = np.stack((x, y), axis=0)  # (2, 133)

            crops = None
            if self.data_cfg['crop_two_hand']:
                crops,missing_wrists_left,missing_wrists_right = crop_hand(frame,keypoints,self.data_cfg['WRIST_DELTA'],self.data_cfg['SHOULDER_DIST_EPSILON'],
                                                                       self.transform,len(clip),missing_wrists_left,missing_wrists_right)
            else:
                crops = self.transform(frame)
            clip.append(crops)

            # Let's say the first frame has a pose flow of 0
            frame_index_poseflow = frame_index
            video_id = name.replace(".mp4", "")
            poseflow_data = self._poseflow_h5[video_id]['poseflow']  # (T-1, 133, 2)
            if frame_index_poseflow > 0 and (frame_index_poseflow - 1) < poseflow_data.shape[0]:
                poseflow = poseflow_data[frame_index_poseflow - 1].copy()  # (133, 2)
                poseflow[:, 0] /= math.pi
            else:
                poseflow = np.zeros((133, 2))

            pose_transform = Compose(
                DeleteFlowKeypoints(list(range(11, 91))),  # delete foot+face → keep body[0:11] + hands[91:133] = 53 kp
                ToFloatTensor()
            )
            poseflow = pose_transform(poseflow).view(-1)
            poseflow_clip.append(poseflow)

            # Hand keypoints from HDF5
            hand_kp_data = self._hand_kp_h5[video_id]['hand_kp']  # (T, 46, 2)
            if frame_index < hand_kp_data.shape[0]:
                handkp_frame = hand_kp_data[frame_index].copy()  # (46, 2)
            else:
                handkp_frame = np.zeros((46, 2))

            # Apply transformations to handflow data
            handkp_frame = self.transform_handflow(handkp_frame)
            handkp_clip.append(handkp_frame)

        clip = torch.stack(clip,dim = 0)
        poseflow = torch.stack(poseflow_clip, dim=0)
        # Stack handflow frames into a tensor along the time dimension
        handkp = torch.stack(handkp_clip, dim=1)  # shape: [C, T, V]
        # Add the M dimension (number of persons), which is 1 in this case
        handkp = handkp.unsqueeze(-1)  # shape: [C, T, V, M]
        return clip,poseflow,handkp

    def read_videos(self,center,left,right):
        index_setting = self.data_cfg['transform_cfg'].get('index_setting', ['consecutive','pad','central','pad'])
        # 
        vlen1,c_width,c_height = self.count_frames(os.path.join(self.base_url,'videos',center))
        vlen2,l_width,l_height = self.count_frames(os.path.join(self.base_url,'videos',left))
        vlen3,r_width,r_height = self.count_frames(os.path.join(self.base_url,'videos',right))

       
        min_vlen = min(vlen1,min(vlen2,vlen3))
        max_vlen = max(vlen1,max(vlen2,vlen3))
        if max_vlen - min_vlen < 10:
            selected_index, pad = get_selected_indexs(min_vlen - 3,self.data_cfg['num_output_frames'],self.is_train,index_setting,temporal_stride=self.data_cfg['temporal_stride'])
            
            if pad is not None:
                selected_index  = pad_index(selected_index,pad).tolist()
        
            center_video,center_pf,center_kp = self.read_one_view(center,selected_index,width=c_width,height=c_height)
            
            left_video,left_pf,left_kp = self.read_one_view(left,selected_index,width=l_width,height=l_height)

            right_video,right_pf,right_kp = self.read_one_view(right,selected_index,width=r_width,height=r_height)
        else:
            selected_index, pad = get_selected_indexs(vlen1 - 3,self.data_cfg['num_output_frames'],self.is_train,index_setting,temporal_stride=self.data_cfg['temporal_stride'])
            
            if pad is not None:
                selected_index  = pad_index(selected_index,pad).tolist()

            center_video,center_pf,center_kp = self.read_one_view(center,selected_index,width=c_width,height=c_height)

            selected_index, pad = get_selected_indexs(vlen2 - 3,self.data_cfg['num_output_frames'],self.is_train,index_setting,temporal_stride=self.data_cfg['temporal_stride'])
            
            if pad is not None:
                selected_index  = pad_index(selected_index,pad).tolist()
            
            
            left_video,left_pf,left_kp = self.read_one_view(left,selected_index,width=l_width,height=l_height)

            selected_index, pad = get_selected_indexs(vlen3-3,self.data_cfg['num_output_frames'],self.is_train,index_setting,temporal_stride=self.data_cfg['temporal_stride'])
            
            if pad is not None:
                selected_index  = pad_index(selected_index,pad).tolist()

            right_video,right_pf,right_kp = self.read_one_view(right,selected_index,width=r_width,height=r_height)

       

        return center_video,center_pf,center_kp,left_video,left_pf,left_kp,right_video,right_pf,right_kp
        # return 1


    # def __getitem__(self, idx):
    #     self.transform.randomize_parameters()
       
    #     center,left,right,label = self.train_labels.iloc[idx].values
       
    #     center_video,center_pf,center_kp,left_video,left_pf,left_kp,right_video,right_pf,right_kp = self.read_videos(center,left,right)
        
    #     return center_video,center_pf,center_kp,left_video,left_pf,left_kp,right_video,right_pf,right_kp,torch.tensor(label)

    def __getitem__(self, idx):
        self.transform.randomize_parameters()

        center, left, right, _id, label = self.train_labels.iloc[idx].values
        center_video, center_pf, center_kp, left_video, left_pf, left_kp, right_video, right_pf, right_kp = self.read_videos(center, left, right)

        if self.is_train:
            left_data = {'rgb': left_video, 'kp': left_kp, 'pf': left_pf}
            center_data = {'rgb': center_video, 'kp': center_kp, 'pf': center_pf}
            right_data = {'rgb': right_video, 'kp': right_kp, 'pf': right_pf}
            
            for aug in self.global_augs:
                left_data, center_data, right_data = aug.apply_synced(left_data, center_data, right_data)
                
            left_data, center_data, right_data = self.synced_spatial(left_data, center_data, right_data)
            
            for aug in self.per_view_augs:
                left_data = aug(left_data)
                center_data = aug(center_data)
                right_data = aug(right_data)

            left_video, left_kp, left_pf = left_data['rgb'], left_data['kp'], left_data['pf']
            center_video, center_kp, center_pf = center_data['rgb'], center_data['kp'], center_data['pf']
            right_video, right_kp, right_pf = right_data['rgb'], right_data['kp'], right_data['pf']

        # Chọn một số ngẫu nhiên từ 1 đến 1000
        random_number = np.random.randint(1, 1001)

        # Kiểm tra điều kiện với missing_rate
        if not (random_number / 1000 > self.data_cfg.get('center_missing_rates', 0)):
            center_video[:] = 0
            center_pf[:] = 0
            center_kp[:, :, :] = 0

        if not (random_number / 1000 > self.data_cfg.get('left_missing_rates', 0)):
            left_video[:] = 0
            left_pf[:] = 0
            left_kp[:, :, :] = 0
            
        if not (random_number / 1000 > self.data_cfg.get('right_missing_rates', 0)):
            right_video[:] = 0
            right_pf[:] = 0
            right_kp[:, :, :] = 0

        return center_video, center_pf, center_kp, left_video, left_pf, left_kp, right_video, right_pf, right_kp, torch.tensor(label)

    
    def __len__(self):
        return len(self.train_labels)
