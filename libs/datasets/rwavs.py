"""
Code of "AV-Cloud: Spatial Audio Rendering Through Audio-Visual Cloud Splatting" 

Copyright (c) 2023-2024 University of Washington. 

Developed in UW NeuroAI Lab by Mingfei Chen (lasiafly@uw.edu).
"""



import json
import os
import pickle
import sys

import imageio
import librosa
import numpy as np
import torch
import torchaudio
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from configs import cfg
from libs.datasets.scene import Scene
from libs.datasets.scene.gaussian_model import GaussianModel


class SceneDataset(torch.utils.data.Dataset):
    def __init__(self,
                 cfg,
                 data_root,
                 split='train', 
                 scene=None,
                 gaussian=None
                 ):
        super(SceneDataset, self).__init__()
        self.data_root = data_root    
        self.cfg = cfg    
        video_name = cfg.dataset.video.split('_')[-1]
        if gaussian is not None and scene is not None:
            self.gaussians = gaussian
            self.scene = scene
        else:
            self.gaussians = GaussianModel(3)
            self.scene = Scene(cfg.dataset.data_root+f'/{video_name}/', f'{cfg.dataset.data_root}/{video_name}/', self.gaussians, sh_degree=3, N_points=cfg.dataset.N_points)
            
        cam_json_path = f'{cfg.dataset.data_root}/{video_name}/gs_cameras.json'
        with open(cam_json_path, 'r') as file:
            cam_data = json.load(file)

        self.cam_dict = {}
        cam_dict = {}
        for cam_item in cam_data:
            cur_name = cam_item["img_name"]
            cam_center = np.array(cam_item["position"]).reshape(-1)
            cam_rotation = np.array(cam_item["rotation"]).reshape(-1)
            cam_dict[f'frames/{cur_name}.png'] = np.hstack([cam_center, cam_rotation])

        self.path_list = []
        with open(f'{data_root}/{video_name}/transforms_scale_{split}.json', 'r') as file:
            cur_cam_dict = json.load(file)["camera_path"]
            for item in cur_cam_dict:
                cur_key = item['file_path']
                idx = int(cur_key[:-4].split('/')[-1])

                if cur_key not in cam_dict.keys():
                    continue

                self.path_list.append((video_name, idx))
                self.cam_dict[cur_key] = cam_dict[cur_key]

        self.split = split

        audio_len = self.cfg.dataset.sr
        self.sr = audio_len
 
        audio_list_path = f'{data_root}/{video_name}/22050_audio_{split}_48k.pkl'
        if not os.path.exists(audio_list_path):
            audio_path = os.path.join(self.data_root, video_name, 'binaural_syn_re.wav')
            source_audio_path = os.path.join(self.data_root, video_name, 'source_syn_re.wav')
            full_gt_audio, audio_rate = librosa.load(audio_path , sr=self.sr, mono=False)
            source_full_gt_audio, audio_rate = librosa.load(source_audio_path , sr=self.sr, mono=False)
            self.audio_list = {}
            self.audio_list[video_name] = {}
            self.audio_list[video_name]['gt'] = full_gt_audio
            self.audio_list[video_name]['source'] = source_full_gt_audio
            with open(audio_list_path, 'wb') as file:
                pickle.dump(self.audio_list, file)
        else:
            with open(audio_list_path, 'rb') as file:
                self.audio_list = pickle.load(file)


        print(f'{split} dataset len: {len(self.path_list)}')



    def __getitem__(self, index):
        video_name, start_idx = self.path_list[index]
        audio_len = self.sr
        key = os.path.join('frames', f'{str(start_idx).zfill(5)}.png')
        cam_pose = torch.from_numpy(np.array(self.cam_dict[key])).float()
        ori_start_idx = start_idx
        
        gt_audio = torch.zeros(2, audio_len)
        cur_gt_audio = torch.from_numpy(self.audio_list[video_name]['gt'][..., (start_idx-1)*audio_len:start_idx*audio_len])
        gt_audio[..., :cur_gt_audio.shape[-1]] = cur_gt_audio.float()

        source_gt_audio = torch.zeros(2, audio_len)
        source_cur_gt_audio = torch.from_numpy(self.audio_list[video_name]['source'][..., (start_idx-1)*audio_len:start_idx*audio_len])
        source_gt_audio[..., :source_cur_gt_audio.shape[-1]] = source_cur_gt_audio.float()
       
        return ori_start_idx, cam_pose, gt_audio, source_gt_audio

    def __len__(self):
        return len(self.path_list)



def build_dataset(cfg):
    train_dataset = SceneDataset(cfg, cfg.dataset.data_root, split='train')
    eval_dataset = SceneDataset(cfg, cfg.dataset.data_root, split='val', scene=train_dataset.scene, gaussian=train_dataset.gaussians)
    return train_dataset, eval_dataset

def build_single_dataset(cfg, split):
    single_dataset = SceneDataset(cfg, cfg.dataset.data_root, split=split)
    return single_dataset

