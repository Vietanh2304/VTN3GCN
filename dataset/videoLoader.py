import os, numpy as np
from utils.zipreader import ZipReader
import io, torch, torchvision
from PIL import Image



 

def get_selected_indexs(vlen, num_frames=64, is_train=True, setting=['consecutive', 'pad', 'central', 'pad'],temporal_stride = 2):
    pad = None  #pad denotes the number of padding frames
    assert len(setting) == 4
    # denote train > 64, test > 64, test < 64
    train_p, train_m, test_p, test_m = setting
    if train_p == 'fusion': 
        train_p = np.random.choice(['consecutive', 'random','segment','center_stride'])
    assert train_p in ['consecutive', 'random','segment','center_stride']
    assert train_m in ['pad']
    assert test_p in ['central', 'start', 'end','segment','center_stride']
    assert test_m in ['pad', 'start_pad', 'end_pad']
    if num_frames > 0:
        assert num_frames%4 == 0
        if is_train:
            if vlen > num_frames:
                
                if train_p == 'consecutive' or train_p == "segment":
                    data_chunks = np.array_split(range(vlen), num_frames)
                    random_elements = np.array([np.random.choice(chunk) for chunk in data_chunks])
                    selected_index = sorted(random_elements)
                elif train_p == 'center_stride':
                    frame_start = (vlen - num_frames) // (2 * temporal_stride)
                    frame_end = frame_start + num_frames * temporal_stride
                    if frame_start < 0:
                        frame_start = 0
                    if frame_end > vlen:
                        frame_end = vlen
                    selected_index = list(range(frame_start, frame_end, temporal_stride))
                    while len(selected_index) < num_frames:
                        selected_index.append(selected_index[-1])
                    selected_index = np.array(selected_index)
                elif train_p == 'random':
                    # random sampling
                    selected_index = np.arange(vlen)
                    np.random.shuffle(selected_index)
                    selected_index = selected_index[:num_frames]  #to make the length equal to that of no drop
                    selected_index = sorted(selected_index)
                else:
                    selected_index = np.arange(0, vlen)
            elif vlen < num_frames:
                if train_m == 'pad':
                    remain = num_frames - vlen
                    selected_index = np.arange(0, vlen)
                    pad_left = np.random.randint(0, remain, 1)[0]
                    pad_right = remain - pad_left
                    pad = (pad_left, pad_right)
                else:
                    selected_index = np.arange(0, vlen)
            else:
                selected_index = np.arange(0, vlen)
        
        else:
            if vlen >= num_frames:
                start = 0
                if test_p == 'central':
                    start = (vlen - num_frames) // 2
                    selected_index = np.arange(start, start+num_frames)
                elif test_p == 'center_stride':
                    frame_start = (vlen - num_frames) // (2 * temporal_stride)
                    frame_end = frame_start + num_frames * temporal_stride
                    if frame_start < 0:
                        frame_start = 0
                    if frame_end > vlen:
                        frame_end = vlen
                    selected_index = list(range(frame_start, frame_end, temporal_stride))
                    while len(selected_index) < num_frames:
                        selected_index.append(selected_index[-1])
                    selected_index = np.array(selected_index)
                elif test_p == 'start':
                    start = 0
                    selected_index = np.arange(start, start+num_frames)
                elif test_p == 'end':
                    start = vlen - num_frames
                    selected_index = np.arange(start, start+num_frames)
                elif test_p == "segment":
                    # selected_index = []
                    # segment_length = int(vlen / num_frames)
                    # mod = vlen - segment_length*num_frames
                    # # Duyệt qua từng segment
                    # for i in range(num_frames):
                    #     # Tìm vị trí frame bắt đầu của segment
                    #     start_frame = i * segment_length
                    #     # Tìm vị trí frame kết thúc của segment
                    #     end_frame = (i + 1) * segment_length

                    #     # Chọn ngẫu nhiên 1 frame trong segment
                    #     random_frame_index = random.randint(start_frame, end_frame - 1)
                    #     selected_index.append(random_frame_index)
                    # # balance the index of samples at  the start and end of video
                    # selected_index = (np.array(selected_index) + random.randint(0, mod)).tolist()
                    data_chunks = np.array_split(range(vlen), num_frames)
                    random_elements = np.array([np.random.choice(chunk) for chunk in data_chunks])
                    selected_index = sorted(random_elements)
                else: 
                    selected_index = np.arange(start, start+num_frames)
            else:
                remain = num_frames - vlen
                selected_index = np.arange(0, vlen)
                if test_m == 'pad':
                    pad_left = remain // 2
                    pad_right = remain - pad_left
                    pad = (pad_left, pad_right)
                elif test_m == 'start_pad':
                    pad_left = 0
                    pad_right = remain - pad_left
                    pad = (pad_left, pad_right)
                elif test_m == 'end_pad':
                    pad_left = remain
                    pad_right = remain - pad_left
                    pad = (pad_left, pad_right)
                else:
                    selected_index = np.arange(0, vlen)
    else:
        # for statistics
        selected_index = np.arange(vlen)

    return selected_index, pad


def pil_list_to_tensor(pil_list, int2float=True):
    func = torchvision.transforms.PILToTensor()
    tensors = [func(pil_img) for pil_img in pil_list]
    #list of C H W
    tensors = torch.stack(tensors, dim=0)
    if int2float:
        tensors = tensors/255
    return tensors #T,C,H,W


def pad_array(array, l_and_r):
    left, right = l_and_r
    if left > 0:
        pad_img = array[0]
        pad = np.tile(np.expand_dims(pad_img, axis=0), tuple([left]+[1]*(len(array.shape)-1)))
        array = np.concatenate([pad, array], axis=0)
    if right > 0:
        pad_img = array[-1]
        pad = np.tile(np.expand_dims(pad_img, axis=0), tuple([right]+[1]*(len(array.shape)-1)))
        array = np.concatenate([array, pad], axis=0)
    return array

def pad_index(index_arr, l_and_r) :
    left, right = l_and_r
    index_arr = index_arr.tolist()
    if len(index_arr) == 0:
        # Video too short (vlen <= 3): use frame 0 for all positions
        index_arr = [0] * (left + right)
    else:
        index_arr = left*[index_arr[0]] + index_arr + right*[index_arr[-1]]
    return np.array(index_arr)
    
def load_video(zip_file, name, vlen, num_frames, dataset_name, is_train, 
                index_setting=['consecutive', 'pad', 'central', 'pad'], temp_scale=[1.0,1.0], ori_vfile=''):
    if 'WLASL' in dataset_name:
        vlen = vlen - 2  # a bug in lintel when load .mp4, by yutong
    selected_index, pad = get_selected_indexs(vlen, num_frames, is_train, index_setting)

    if 'WLASL' in dataset_name:
        video_file = '{:s}.mp4'.format(name)
        path = zip_file+'@'+video_file
        video_byte = ZipReader.read(path)
        video_arrays = _load_frame_nums_to_4darray(video_byte, selected_index) #T,H,W,3
    elif 'MSASL' in dataset_name or 'NMFs-CSL' in dataset_name:
        video_arrays = read_jpg(zip_file, dataset_name, selected_index, vlen, ori_vfile)
    
    # pad
    if pad is not None:
        video_arrays = pad_array(video_arrays, pad)
    return video_arrays, selected_index, pad


def load_batch_video(zip_file, names, vlens, dataset_name, is_train, 
                    num_output_frames=64, name2keypoint=None, index_setting=['consecutive','pad','central','pad'], 
                    temp_scale=[1.0,1.0], ori_video_files=[], from64=False):
    #load_video and keypoints, used in collate_fn
    sgn_videos, sgn_keypoints = [], []
    batch_videos, batch_keypoints = [], []
    for name, vlen, ori_vfile in zip(names, vlens, ori_video_files):
        video, selected_index, pad = load_video(zip_file, name, vlen, num_output_frames, dataset_name, is_train, index_setting, temp_scale, ori_vfile)
        # video = torch.tensor(video).to(torch.uint8)
        video = torch.tensor(video).float()  # T,H,W,C
        video = video.permute(0, 3, 1, 2)    # T,C,H,W
        
        if 'NMFs-CSL' in dataset_name:
            video = torchvision.transforms.functional.resize(video, [256, 256])

        if is_train:
            rgb_augment = torchvision.transforms.Compose([
                                torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.3),
                torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3)
            ])
            video = rgb_augment(video)
            
        video = torchvision.transforms.functional.resize(video, [256, 256]).permute(0, 2, 3, 1) # test 
        video /= 255
        batch_videos.append(video) #wo transformed!!
        
        if name2keypoint != None:
            kps = name2keypoint[name][selected_index,:,:]
            if pad is not None:
                kps = pad_array(kps, pad)

            if is_train:
                noise = np.random.normal(0, 0.01, size=kps.shape)
                kps = kps + noise

            batch_keypoints.append(torch.from_numpy(kps).float()) # T,N,3
        else:
            batch_keypoints.append(None)

    batch_videos = torch.stack(batch_videos, dim=0).permute(0,1,4,2,3) #B,T,C,H,W for spatial augmentation

    if name2keypoint != None:
        batch_keypoints = torch.stack(batch_keypoints, dim=0) #B,T,N,3
    else:
        batch_keypoints = None
    
    sgn_videos.append(batch_videos)
    sgn_keypoints.append(batch_keypoints)
    
    if from64:
        #32-frame counterpart
        if is_train:
            st = np.random.randint(0, num_output_frames//2+1, 1)[0]
        else:
            st = num_output_frames//4
        end = st + num_output_frames//2
        
        sgn_videos.append(sgn_videos[-1][:, st:end, ...])
        if sgn_keypoints[-1] is None:
            sgn_keypoints.append(None)
        else:
            sgn_keypoints.append(sgn_keypoints[-1][:, st:end, ...])
    
    return sgn_videos, sgn_keypoints


def is_zip_path(img_or_path):
    """judge if this is a zip path"""
    return '.zip@' in img_or_path


