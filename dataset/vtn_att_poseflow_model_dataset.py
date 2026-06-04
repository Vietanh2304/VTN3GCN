import json
import math
import os
import h5py
from argparse import ArgumentParser
import numpy as np
from decord import VideoReader
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from utils.video_augmentation import DeleteFlowKeypoints, ToFloatTensor, Compose
from dataset.videoLoader import get_selected_indexs, pad_index
import cv2
from utils.video_augmentation import *
from dataset.utils import crop_hand
import glob
from PIL import Image

class VTN_ATT_PF_Dataset(Dataset):
    def __init__(self, base_url, split, dataset_cfg, train_labels=None, **kwargs):
        if train_labels is None:
            if dataset_cfg['dataset_name'] in ["VN_SIGN", "AUTSL", "VSL400"]:
                if "merged_dataset" in base_url:
                    csv_path = os.path.normpath(os.path.join(base_url, "..", f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"))
                else:
                    csv_path = os.path.normpath(os.path.join(base_url, f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"))
                
                print("Label Path Found: ", csv_path)
                self.train_labels = pd.read_csv(csv_path, sep=',')
            else:
                raise ValueError(f"Dataset name '{dataset_cfg['dataset_name']}' chưa được định nghĩa luồng đọc nhãn!")
        else:
            print("Use labels from K-Fold")
            self.train_labels = train_labels
            
        print(split, len(self.train_labels))
        self.split = split
        if split == 'train':
            self.is_train = True
        else:
            self.is_train = False
        self.base_url = base_url
        self.data_cfg = dataset_cfg
        self.data_name = dataset_cfg['dataset_name']
        self.transform = self.build_transform(split)
        if self.data_name == "VSL400":
            print(f"--> Đang kết nối đến file H5 Wholebody: {self.data_cfg['wholebody_h5_dir']}")
            self.wholebody_h5 = h5py.File(self.data_cfg['wholebody_h5_dir'], 'r')
            
            if 'poseflow_h5_dir' in self.data_cfg:
                print(f"--> Đang kết nối đến file H5 Poseflow: {self.data_cfg['poseflow_h5_dir']}")
                self.poseflow_h5 = h5py.File(self.data_cfg['poseflow_h5_dir'], 'r')

    def build_transform(self, split):
        if split == 'train':
            print("Build train transform")
            transform = Compose(
                                Scale(self.data_cfg['vid_transform']['IMAGE_SIZE'] * 8 // 7),
                                MultiScaleCrop((self.data_cfg['vid_transform']['IMAGE_SIZE'], self.data_cfg['vid_transform']['IMAGE_SIZE']), scales),
                                RandomVerticalFlip(),
                                RandomRotate(p=0.3),
                                RandomShear(0.3, 0.3, p=0.3),
                                Salt(p=0.3),
                                GaussianBlur(sigma=1, p=0.3),
                                ColorJitter(0.5, 0.5, 0.5, p=0.3),
                                ToFloatTensor(), PermuteImage(),
                                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'], self.data_cfg['vid_transform']['NORM_STD_IMGNET']))
        else:
            print("Build test/val transform")
            transform = Compose(
                                Scale(self.data_cfg['vid_transform']['IMAGE_SIZE'] * 8 // 7), 
                                CenterCrop(self.data_cfg['vid_transform']['IMAGE_SIZE']), 
                                ToFloatTensor(),
                                PermuteImage(),
                                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'], self.data_cfg['vid_transform']['NORM_STD_IMGNET']))
        return transform
    
    def count_frames(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video file: {video_path}")
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            cap.release()
            return total_frames, width, height
        except Exception as e:
            print(f"Error reading video {video_path}: {e}")
            return None, None, None

    def read_videos(self, name):
        index_setting = self.data_cfg['transform_cfg'].get('index_setting', ['consecutive', 'pad', 'central', 'pad'])

        if self.data_name == "VSL400":
            path = f'{self.base_url}/{name}'
        else:
            path = f'{self.base_url}/videos/{name}'

        vlen, width, height = self.count_frames(path)
        if vlen is None or vlen <= 5:
            vlen = 60  

        selected_index, pad = get_selected_indexs(vlen - 5, self.data_cfg['num_output_frames'], self.is_train,
                                                  index_setting, temporal_stride=self.data_cfg['temporal_stride'])
        if pad is not None:
            selected_index = pad_index(selected_index, pad).tolist()

        try:
            vr = VideoReader(path, width=320, height=256)
            frames = vr.get_batch(selected_index).asnumpy()
        except Exception as e:
            frames = np.zeros((len(selected_index), 256, 320, 3), dtype=np.uint8)

        poseflow_clip = []
        clip = []
        missing_wrists_left = []
        missing_wrists_right = []

        video_key = os.path.basename(name).replace(".mp4", "")

        for frame, frame_index in zip(frames, selected_index):
            crop_keypoints = np.zeros((2, 26))
            
            if self.data_cfg['crop_two_hand']:
                if self.data_name == "VSL400" and str(video_key) in self.wholebody_h5:
                    node = self.wholebody_h5[str(video_key)]
                    if isinstance(node, h5py.Group):
                        first_sub_key = list(node.keys())[0]
                        video_data = np.array(node[first_sub_key])
                    else:
                        video_data = np.array(node)
                        
                    safe_index = min(frame_index, len(video_data) - 1)
                    raw_kp = video_data[safe_index]
                    
                    # Bảo hiểm: Nếu mảng trích xuất thô bị dị thường, không đủ hình dạng [26, >=2]
                    if len(raw_kp.shape) == 2 and raw_kp.shape[0] >= 26:
                        x = 320 * raw_kp[:, 0] / width
                        y = 256 * raw_kp[:, 1] / height
                        crop_keypoints = np.stack((x, y), axis=0)
                    else:
                        crop_keypoints = np.zeros((2, 26))
                else:
                    n = 0
                    found = False
                    while frame_index - n >= 0:
                        current_index = frame_index - n
                        kp_path = os.path.join(self.base_url, 'poses', name.replace(".mp4", ""), f"{name.replace('.mp4', '')}_{current_index:06d}_keypoints.json")
                        if os.path.exists(kp_path):
                            with open(kp_path, 'r') as keypoints_file:
                                value = json.load(keypoints_file)
                                raw_kp = np.array(value['pose_threshold_02'])
                                if len(raw_kp.shape) == 2 and raw_kp.shape[0] >= 26:
                                    x = 320 * raw_kp[:, 0] / width
                                    y = 256 * raw_kp[:, 1] / height
                                    crop_keypoints = np.stack((x, y), axis=0)
                                else:
                                    crop_keypoints = np.zeros((2, 26))
                            found = True
                            break
                        else:
                            n += 1

            # Ép thêm một tầng bảo hiểm cuối cùng trước khi đưa vào hàm của utils.py
            if crop_keypoints.shape != (2, 26):
                crop_keypoints = np.zeros((2, 26))

            crops = None
            if self.data_cfg['crop_two_hand']:
                crops, missing_wrists_left, missing_wrists_right = crop_hand(frame, crop_keypoints, self.data_cfg['WRIST_DELTA'], self.data_cfg['SHOULDER_DIST_EPSILON'],
                                                                             self.transform, len(clip), missing_wrists_left, missing_wrists_right)
            else:
                crops = self.transform(frame)
            clip.append(crops)

            poseflow = np.zeros((135, 2))
            if frame_index > 0:
                if self.data_name == "VSL400" and str(video_key) in self.poseflow_h5:
                    f_node = self.poseflow_h5[str(video_key)]
                    if isinstance(f_node, h5py.Group):
                        first_f_key = list(f_node.keys())[0]
                        flow_data = np.array(f_node[first_f_key])
                    else:
                        flow_data = np.array(f_node)
                        
                    safe_f_index = min(frame_index, len(flow_data) - 1)
                    poseflow = flow_data[safe_f_index]
                    poseflow[:, 0] /= math.pi
                else:
                    frame_index_poseflow = frame_index
                    full_path = os.path.join(self.base_url, 'poseflow', name.replace(".mp4", ""), 'flow_{:05d}.npy'.format(frame_index_poseflow))
                    while not os.path.isfile(full_path) and frame_index_poseflow > 0:
                        frame_index_poseflow -= 1
                        full_path = os.path.join(self.base_url, 'poseflow', name.replace(".mp4", ""), 'flow_{:05d}.npy'.format(frame_index_poseflow))
                    if os.path.isfile(full_path):
                        poseflow = np.load(full_path)
                        poseflow[:, 0] /= math.pi

            pose_transform = Compose(DeleteFlowKeypoints(list(range(114, 115))),
                                     DeleteFlowKeypoints(list(range(19, 94))),
                                     DeleteFlowKeypoints(list(range(11, 17))),
                                     ToFloatTensor())

            poseflow = pose_transform(poseflow).view(-1)
            
            target_size = 106
            if poseflow.size(0) < target_size:
                padding = torch.zeros(target_size - poseflow.size(0), dtype=poseflow.dtype)
                poseflow = torch.cat((poseflow, padding), dim=0)
            elif poseflow.size(0) > target_size:
                poseflow = poseflow[:target_size]
                
            poseflow_clip.append(poseflow)
            
        clip = torch.stack(clip, dim=0)
        poseflow = torch.stack(poseflow_clip, dim=0)
        return clip, poseflow

    def __getitem__(self, idx):
        self.transform.randomize_parameters()
        data = self.train_labels.iloc[idx].values
        name, label = data[0], data[1]
        clip, poseflow = self.read_videos(name)
        return clip, poseflow, torch.tensor(label)

    def __len__(self):
        return len(self.train_labels)

class VTN_GCN_Dataset(Dataset):
    def __init__(self, base_url, split, dataset_cfg, train_labels=None, **kwargs):
        if train_labels is None:
            if dataset_cfg['dataset_name'] in ["VN_SIGN", "AUTSL", "VSL400"]:
                if "merged_dataset" in base_url:
                    csv_path = os.path.normpath(os.path.join(base_url, "..", f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"))
                else:
                    csv_path = os.path.normpath(os.path.join(base_url, f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"))
                
                print("Label Path Found: ", csv_path)
                self.train_labels = pd.read_csv(csv_path, sep=',')
            else:
                raise ValueError(f"Dataset name '{dataset_cfg['dataset_name']}' chưa được định nghĩa luồng đọc nhãn!")
        else:
            print("Use labels from K-Fold")
            self.train_labels = train_labels
            
        print(split, len(self.train_labels))
        self.split = split
        if split == 'train':
            self.is_train = True
        else:
            self.is_train = False
        self.base_url = base_url
        self.data_cfg = dataset_cfg
        self.data_name = dataset_cfg['dataset_name']
        self.transform = self.build_transform(split)
    
        if self.data_name == "VSL400":
            print(f"--> [GCN] Kết nối H5 Wholebody: {self.data_cfg['wholebody_h5_dir']}")
            self.wholebody_h5 = h5py.File(self.data_cfg['wholebody_h5_dir'], 'r')
            if 'poseflow_h5_dir' in self.data_cfg:
                self.poseflow_h5 = h5py.File(self.data_cfg['poseflow_h5_dir'], 'r')
            if 'hand_kp_h5_dir' in self.data_cfg:
                self.hand_kp_h5 = h5py.File(self.data_cfg['hand_kp_h5_dir'], 'r')

    def transform_handkp(self, handkp):
        handkp_tensor = torch.tensor(handkp, dtype=torch.float32).transpose(0, 1)
        return handkp_tensor
        
    def build_transform(self, split):
        if split == 'train':
            print("Build train transform")
            transform = Compose(
                                Scale(self.data_cfg['vid_transform']['IMAGE_SIZE'] * 8 // 7),
                                MultiScaleCrop((self.data_cfg['vid_transform']['IMAGE_SIZE'], self.data_cfg['vid_transform']['IMAGE_SIZE']), scales),
                                RandomVerticalFlip(),
                                RandomRotate(p=0.3),
                                RandomShear(0.3, 0.3, p=0.3),
                                Salt(p=0.3),
                                GaussianBlur(sigma=1, p=0.3),
                                ColorJitter(0.5, 0.5, 0.5, p=0.3),
                                ToFloatTensor(), PermuteImage(),
                                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'], self.data_cfg['vid_transform']['NORM_STD_IMGNET']))
        else:
            print("Build test/val transform")
            transform = Compose(
                                Scale(self.data_cfg['vid_transform']['IMAGE_SIZE'] * 8 // 7), 
                                CenterCrop(self.data_cfg['vid_transform']['IMAGE_SIZE']), 
                                ToFloatTensor(),
                                PermuteImage(),
                                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'], self.data_cfg['vid_transform']['NORM_STD_IMGNET']))
        return transform
    
    def count_frames(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video file: {video_path}")
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return total_frames, width, height
        except Exception as e:
            print(f"Error reading video {video_path}: {e}")
            return None, None, None

    def read_videos(self, name):
        index_setting = self.data_cfg['transform_cfg'].get('index_setting', ['consecutive', 'pad', 'central', 'pad'])
       
        if self.data_name == "VSL400":
            path = f'{self.base_url}/{name}'   
        else:
            path = f'{self.base_url}/videos/{name}'  
        vlen, width, height = self.count_frames(path)
       
        if vlen is None or vlen <= 5:
            vlen = 60 
            
        selected_index, pad = get_selected_indexs(vlen - 5, self.data_cfg['num_output_frames'], self.is_train, index_setting, temporal_stride=self.data_cfg['temporal_stride'])
        if pad is not None:
            selected_index = pad_index(selected_index, pad).tolist()
            
        try:
            vr = VideoReader(path, width=320, height=256)
            frames = vr.get_batch(selected_index).asnumpy()
        except Exception as e:
            frames = np.zeros((len(selected_index), 256, 320, 3), dtype=np.uint8)

        poseflow_clip = []
        clip = []
        handkp_clip = []
        missing_wrists_left = []
        missing_wrists_right = []

        video_key = os.path.basename(name).replace(".mp4", "")

        for frame, frame_index in zip(frames, selected_index):
            crop_keypoints = np.zeros((2, 26))
            
            if self.data_cfg['crop_two_hand']:
                if self.data_name == "VSL400" and str(video_key) in self.wholebody_h5:
                    node = self.wholebody_h5[str(video_key)]
                    if isinstance(node, h5py.Group):
                        first_sub_key = list(node.keys())[0]
                        video_data = np.array(node[first_sub_key])
                    else:
                        video_data = np.array(node)
                        
                    safe_index = min(frame_index, len(video_data) - 1)
                    raw_kp = video_data[safe_index]
                    if len(raw_kp.shape) == 2 and raw_kp.shape[0] >= 26:
                        x = 320 * raw_kp[:, 0] / width
                        y = 256 * raw_kp[:, 1] / height
                        crop_keypoints = np.stack((x, y), axis=0)
                    else:
                        crop_keypoints = np.zeros((2, 26))
                else:
                    n = 0
                    found = False
                    while frame_index - n >= 0:
                        current_index = frame_index - n
                        kp_path = os.path.join(self.base_url, 'poses', name.replace(".mp4", ""), f"{name.replace('.mp4', '')}_{current_index:06d}_keypoints.json")
                        if os.path.exists(kp_path):
                            with open(kp_path, 'r') as keypoints_file:
                                value = json.load(keypoints_file)
                                raw_kp = np.array(value['pose_threshold_02'])
                                if len(raw_kp.shape) == 2 and raw_kp.shape[0] >= 26:
                                    x = 320 * raw_kp[:, 0] / width
                                    y = 256 * raw_kp[:, 1] / height
                                    crop_keypoints = np.stack((x, y), axis=0)
                                else:
                                    crop_keypoints = np.zeros((2, 26))
                            found = True
                            break
                        else:
                            n += 1

            if crop_keypoints.shape != (2, 26):
                crop_keypoints = np.zeros((2, 26))

            crops = None
            if self.data_cfg['crop_two_hand']:
                crops, missing_wrists_left, missing_wrists_right = crop_hand(frame, crop_keypoints, self.data_cfg['WRIST_DELTA'], self.data_cfg['SHOULDER_DIST_EPSILON'],
                                                                             self.transform, len(clip), missing_wrists_left, missing_wrists_right)
            else:
                crops = self.transform(frame)
            clip.append(crops)

            poseflow = np.zeros((135, 2))
            if frame_index > 0:
                if self.data_name == "VSL400" and str(video_key) in self.poseflow_h5:
                    f_node = self.poseflow_h5[str(video_key)]
                    if isinstance(f_node, h5py.Group):
                        first_f_key = list(f_node.keys())[0]
                        flow_data = np.array(f_node[first_f_key])
                    else:
                        flow_data = np.array(f_node)
                        
                    safe_f_index = min(frame_index, len(flow_data) - 1)
                    poseflow = flow_data[safe_f_index]
                    poseflow[:, 0] /= math.pi
                else:
                    frame_index_poseflow = frame_index
                    full_path = os.path.join(self.base_url, 'poseflow', name.replace(".mp4", ""), 'flow_{:05d}.npy'.format(frame_index_poseflow))
                    while not os.path.isfile(full_path) and frame_index_poseflow > 0:
                        frame_index_poseflow -= 1
                        full_path = os.path.join(self.base_url, 'poseflow', name.replace(".mp4", ""), 'flow_{:05d}.npy'.format(frame_index_poseflow))
                    if os.path.isfile(full_path):
                        poseflow = np.load(full_path)
                        poseflow[:, 0] /= math.pi

            pose_transform = Compose(DeleteFlowKeypoints(list(range(114, 115))),
                                     DeleteFlowKeypoints(list(range(19, 94))),
                                     DeleteFlowKeypoints(list(range(11, 17))),
                                     ToFloatTensor())
            poseflow = pose_transform(poseflow).view(-1)
            
            target_size = 106
            if poseflow.size(0) < target_size:
                padding = torch.zeros(target_size - poseflow.size(0), dtype=poseflow.dtype)
                poseflow = torch.cat((poseflow, padding), dim=0)
            elif poseflow.size(0) > target_size:
                poseflow = poseflow[:target_size]
                
            poseflow_clip.append(poseflow)

            handkp_frame = np.zeros((46, 2))
            if self.data_name == "VSL400" and hasattr(self, 'hand_kp_h5') and str(video_key) in self.hand_kp_h5:
                h_node = self.hand_kp_h5[str(video_key)]
                if isinstance(h_node, h5py.Group):
                    first_h_key = list(h_node.keys())[0]
                    hand_data = np.array(h_node[first_h_key])
                else:
                    hand_data = np.array(h_node)
                    
                safe_h_index = min(frame_index, len(hand_data) - 1)
                handkp_frame = hand_data[safe_h_index]
            else:
                frame_index_handkp = frame_index
                full_path = os.path.join(self.base_url, 'hand_keypoints', name.replace(".mp4", ""), f'hand_kp_{frame_index_handkp:05d}.npy')
                while not os.path.isfile(full_path) and frame_index_handkp > 0:
                    frame_index_handkp -= 1
                    full_path = os.path.join(self.base_url, 'hand_keypoints', name.replace(".mp4", ""), f'hand_kp_{frame_index_handkp:05d}.npy')
                if os.path.isfile(full_path):
                    handkp_frame = np.load(full_path)

            handkp_frame = self.transform_handkp(handkp_frame)
            handkp_clip.append(handkp_frame)

        clip = torch.stack(clip, dim=0)
        poseflow = torch.stack(poseflow_clip, dim=0)
        handkp = torch.stack(handkp_clip, dim=1) 
        handkp = handkp.unsqueeze(-1) 
        return clip, poseflow, handkp

    def __getitem__(self, idx):
        self.transform.randomize_parameters()
        data = self.train_labels.iloc[idx].values
        name, label = data[0], data[1]
        clip, poseflow, handkp = self.read_videos(name)
        return clip, poseflow, handkp, torch.tensor(label)

    def __len__(self):
        return len(self.train_labels)

class VTN_RGBheat_Dataset(Dataset):
    def __init__(self, base_url, split, dataset_cfg, train_labels=None, **kwargs):
        if train_labels is None:
            if dataset_cfg['dataset_name'] in ["VN_SIGN", "AUTSL", "VSL400"]:
                if "merged_dataset" in base_url:
                    csv_path = os.path.normpath(os.path.join(base_url, "..", f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"))
                else:
                    csv_path = os.path.normpath(os.path.join(base_url, f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"))
                
                print("Label Path Found: ", csv_path)
                self.train_labels = pd.read_csv(csv_path, sep=',')
            else:
                raise ValueError(f"Dataset name '{dataset_cfg['dataset_name']}' chưa được định nghĩa luồng đọc nhãn!")
        else:
            print("Use labels from K-Fold")
            self.train_labels = train_labels

        print(split, len(self.train_labels))
        self.split = split
        if split == 'train':
            self.is_train = True
        else:
            self.is_train = False
        self.base_url = base_url
        self.data_cfg = dataset_cfg
        self.data_name = dataset_cfg['dataset_name']
        self.transform = self.build_transform(split)
        if self.data_name == "VSL400":
            print(f"--> [RGBheat] Kết nối H5 Wholebody: {self.data_cfg['wholebody_h5_dir']}")
            self.wholebody_h5 = h5py.File(self.data_cfg['wholebody_h5_dir'], 'r')
            if 'poseflow_h5_dir' in self.data_cfg:
                self.poseflow_h5 = h5py.File(self.data_cfg['poseflow_h5_dir'], 'r')
            if 'heatmap_h5_dir' in self.data_cfg: 
                self.heatmap_h5 = h5py.File(self.data_cfg['heatmap_h5_dir'], 'r')

    def build_transform(self, split):
        if split == 'train':
            print("Build train transform")
            transform = Compose(
                Scale(self.data_cfg['vid_transform']['IMAGE_SIZE'] * 8 // 7),
                MultiScaleCrop((self.data_cfg['vid_transform']['IMAGE_SIZE'], self.data_cfg['vid_transform']['IMAGE_SIZE']), scales),
                RandomVerticalFlip(),
                ColorJitter(0.5, 0.5, 0.5, p=0.3),
                ToFloatTensor(), PermuteImage(),
                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'], self.data_cfg['vid_transform']['NORM_STD_IMGNET']))
        else:
            print("Build test/val transform")
            transform = Compose(
                Scale(self.data_cfg['vid_transform']['IMAGE_SIZE'] * 8 // 7),
                CenterCrop(self.data_cfg['vid_transform']['IMAGE_SIZE']),
                ToFloatTensor(),
                PermuteImage(),
                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'], self.data_cfg['vid_transform']['NORM_STD_IMGNET']))
        return transform

    def count_frames(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video file: {video_path}")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            cap.release()
            return total_frames, width, height
        except Exception as e:
            print(f"Error reading video {video_path}: {e}")
            return None, None, None

    def read_videos(self, name):
        index_setting = self.data_cfg['transform_cfg'].get('index_setting', ['consecutive', 'pad', 'central', 'pad'])

        if self.data_name == "VSL400":
            path = f'{self.base_url}/{name}'
        else:
            path = f'{self.base_url}/videos/{name}'
            
        vlen, width, height = self.count_frames(path)
        if vlen is None or vlen <= 5:
            vlen = 60  

        selected_index, pad = get_selected_indexs(vlen - 5, self.data_cfg['num_output_frames'], self.is_train,
                                                  index_setting, temporal_stride=self.data_cfg['temporal_stride'])
        if pad is not None:
            selected_index = pad_index(selected_index, pad).tolist()
            
        try:
            vr = VideoReader(path, width=320, height=256)
            frames = vr.get_batch(selected_index).asnumpy()
        except Exception as e:
            frames = np.zeros((len(selected_index), 256, 320, 3), dtype=np.uint8)

        poseflow_clip = []
        clip = []
        heatmap_clip = []
        missing_wrists_left = []
        missing_wrists_right = []

        video_key = os.path.basename(name).replace(".mp4", "")

        for frame, frame_index in zip(frames, selected_index):
            crop_keypoints = np.zeros((2, 26)) 
            
            if self.data_cfg['crop_two_hand']:
                if self.data_name == "VSL400" and str(video_key) in self.wholebody_h5:
                    node = self.wholebody_h5[str(video_key)]
                    if isinstance(node, h5py.Group):
                        first_sub_key = list(node.keys())[0]
                        video_data = np.array(node[first_sub_key])
                    else:
                        video_data = np.array(node)
                        
                    safe_index = min(frame_index, len(video_data) - 1)
                    raw_kp = video_data[safe_index]
                    if len(raw_kp.shape) == 2 and raw_kp.shape[0] >= 26:
                        x = 320 * raw_kp[:, 0] / width
                        y = 256 * raw_kp[:, 1] / height
                        crop_keypoints = np.stack((x, y), axis=0)
                    else:
                        crop_keypoints = np.zeros((2, 26))
                else:
                    n = 0
                    found = False
                    while frame_index - n >= 0:
                        current_index = frame_index - n
                        kp_path = os.path.join(self.base_url, 'poses', name.replace(".mp4", ""), f"{name.replace('.mp4', '')}_{current_index:06d}_keypoints.json")
                        if os.path.exists(kp_path):
                            with open(kp_path, 'r') as keypoints_file:
                                value = json.load(keypoints_file)
                                raw_kp = np.array(value['pose_threshold_02'])
                                if len(raw_kp.shape) == 2 and raw_kp.shape[0] >= 26:
                                    x = 320 * raw_kp[:, 0] / width
                                    y = 256 * raw_kp[:, 1] / height
                                    crop_keypoints = np.stack((x, y), axis=0)
                                else:
                                    crop_keypoints = np.zeros((2, 26))
                            found = True
                            break
                        else:
                            n += 1

            if crop_keypoints.shape != (2, 26):
                crop_keypoints = np.zeros((2, 26))

            crops = None
            if self.data_cfg['crop_two_hand']:
                crops, missing_wrists_left, missing_wrists_right = crop_hand(frame, crop_keypoints,
                                                                             self.data_cfg['WRIST_DELTA'],
                                                                             self.data_cfg['SHOULDER_DIST_EPSILON'],
                                                                             self.transform, len(clip),
                                                                             missing_wrists_left, missing_wrists_right)
            else:
                crops = self.transform(frame)
            clip.append(crops)

            frame_index_heatmap = frame_index  
            heatmap = None
            if frame_index_heatmap > 0:
                if self.data_name == "VSL400" and hasattr(self, 'heatmap_h5') and str(video_key) in self.heatmap_h5:
                    h_node = self.heatmap_h5[str(video_key)]
                    if isinstance(h_node, h5py.Group):
                        first_h_key = list(h_node.keys())[0]
                        heat_data = np.array(h_node[first_h_key])
                    else:
                        heat_data = np.array(h_node)
                        
                    safe_h_index = min(frame_index, len(heat_data) - 1)
                    heatmap = heat_data[safe_h_index]
                    heatmap = self.transform(heatmap)
                else:
                    if self.data_name == "VSL400":
                        view_dir = "front" if "front" in name else ("left" if "left" in name else "right")
                        full_path = os.path.join(self.base_url, 'heatmap', view_dir, video_key, 'heatmap_{:05d}.jpg'.format(frame_index_heatmap))
                    else:
                        full_path = os.path.join(self.base_url, 'heatmap', name.replace(".mp4", ""), 'heatmap_{:05d}.jpg'.format(frame_index_heatmap))
                                                 
                    while not os.path.isfile(full_path):
                        frame_index_heatmap -= 1
                        if frame_index_heatmap <= 0:
                            break
                        if self.data_name == "VSL400":
                            view_dir = "front" if "front" in name else ("left" if "left" in name else "right")
                            full_path = os.path.join(self.base_url, 'heatmap', view_dir, video_key, 'heatmap_{:05d}.jpg'.format(frame_index_heatmap))
                        else:
                            full_path = os.path.join(self.base_url, 'heatmap', name.replace(".mp4", ""), 'heatmap_{:05d}.jpg'.format(frame_index_heatmap))

                    if os.path.isfile(full_path):
                        heatmap = Image.open(full_path).convert('RGB')
                        heatmap = np.array(heatmap)
                        heatmap = self.transform(heatmap)
                    else:
                        heatmap = torch.zeros((3, 224, 224))
            else:
                heatmap = torch.zeros((3, 224, 224))
            heatmap_clip.append(heatmap)

            poseflow = np.zeros((135, 2))
            if frame_index > 0:
                if self.data_name == "VSL400" and str(video_key) in self.poseflow_h5:
                    f_node = self.poseflow_h5[str(video_key)]
                    if isinstance(f_node, h5py.Group):
                        first_f_key = list(f_node.keys())[0]
                        flow_data = np.array(f_node[first_f_key])
                    else:
                        flow_data = np.array(f_node)
                        
                    safe_f_index = min(frame_index, len(flow_data) - 1)
                    poseflow = flow_data[safe_f_index]
                    poseflow[:, 0] /= math.pi
                else:
                    frame_index_poseflow = frame_index
                    if self.data_name == "VSL400":
                        view_dir = "front" if "front" in name else ("left" if "left" in name else "right")
                        full_path = os.path.join(self.base_url, 'poseflow', view_dir, video_key, 'flow_{:05d}.npy'.format(frame_index_poseflow))
                    else:
                        full_path = os.path.join(self.base_url, 'poseflow', name.replace(".mp4", ""), 'flow_{:05d}.npy'.format(frame_index_poseflow))
                                                 
                    while not os.path.isfile(full_path) and frame_index_poseflow > 0:
                        frame_index_poseflow -= 1
                        if self.data_name == "VSL400":
                            view_dir = "front" if "front" in name else ("left" if "left" in name else "right")
                            full_path = os.path.join(self.base_url, 'poseflow', view_dir, video_key, 'flow_{:05d}.npy'.format(frame_index_poseflow))
                        else:
                            full_path = os.path.join(self.base_url, 'poseflow', name.replace(".mp4", ""), 'flow_{:05d}.npy'.format(frame_index_poseflow))

                    if os.path.isfile(full_path):
                        value = np.load(full_path)
                        poseflow = value
                        poseflow[:, 0] /= math.pi

            pose_transform = Compose(DeleteFlowKeypoints(list(range(114, 115))),
                                     DeleteFlowKeypoints(list(range(19, 94))),
                                     DeleteFlowKeypoints(list(range(11, 17))),
                                     ToFloatTensor())
            poseflow = pose_transform(poseflow).view(-1)
            
            target_size = 106
            if poseflow.size(0) < target_size:
                padding = torch.zeros(target_size - poseflow.size(0), dtype=poseflow.dtype)
                poseflow = torch.cat((poseflow, padding), dim=0)
            elif poseflow.size(0) > target_size:
                poseflow = poseflow[:target_size]
                
            poseflow_clip.append(poseflow)

        heatmap = torch.stack(heatmap_clip, dim=0)
        clip = torch.stack(clip, dim=0)
        poseflow = torch.stack(poseflow_clip, dim=0)
        return heatmap, clip, poseflow

    def __getitem__(self, idx):
        self.transform.randomize_parameters()
        data = self.train_labels.iloc[idx].values
        name, label = data[0], data[1]
        heatmap, clip, poseflow = self.read_videos(name)
        return heatmap, clip, poseflow, torch.tensor(label)

    def __len__(self):
        return len(self.train_labels)