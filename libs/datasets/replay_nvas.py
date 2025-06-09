"""
Code of "AV-Cloud: Spatial Audio Rendering Through Audio-Visual Cloud Splatting" 

Copyright (c) 2023-2024 University of Washington. 

Developed in UW NeuroAI Lab by Mingfei Chen (lasiafly@uw.edu).
"""



import json
import os
import pickle
import random
import warnings

import einops
import julius
import librosa
import numpy as np
import torch
import torchaudio
import torchvision
from decord import AudioReader, VideoReader, cpu
from PIL import Image
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, sosfiltfilt
from torch.utils.data import Dataset

from libs.datasets.scene import Scene
from libs.datasets.scene.gaussian_model import GaussianModel

warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter("ignore", UserWarning)



SCENE_SPLITS = {
    'v3': {
        'train': ['SC-1025',  'SC-1032', 'SC-1033',  'SC-1045', 'SC-1059', 'SC-1069', 'SC-1078',
                  'SC-1066',  'SC-1070', 'SC-1076',  'SC-1081', 'SC-1077', 'SC-1035', 
                  'SC-1085', 'SC-1087',  'SC-1101', 'SC-1102', 'SC-1104', 'SC-1106', 'SC-1105', 'SC-1090',
                  'SC-1036', 'SC-1048', 'SC-1049', 'SC-1075', 'SC-1108', 'SC-1047', 'SC-1050'], # 28
        'val': ['SC-1026', 'SC-1054', 'SC-1073', 'SC-1082', 'SC-1092', 'SC-1103'],  # 6
        'test': ['SC-1027', 'SC-1044', 'SC-1052', 'SC-1074', 'SC-1084', 'SC-1093', 'SC-1107']  # 7

    },
}


def to_tensor(v):
    import numpy as np

    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y


def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16, return_second=True):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''

    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)

    if not return_second:
        tau *= fs

    return tau, cc


def read_clip(img_file, split, target_sr):
    video_file = img_file.replace('.png', '.mp4')
    binaural_file = img_file.replace('.png', '.wav')
    binaural_pkl_file = img_file.replace('.png', f'_{target_sr}.pkl')
    if os.path.exists(img_file):
        rgb = np.array(Image.open(img_file))
    else:
        print('Imag file does not exist: ', img_file)
        vr = VideoReader(video_file, ctx=cpu(0))
        rgb = vr[np.random.randint(0, len(vr)) if split == 'train' else len(vr) // 2].asnumpy()

    if os.path.exists(binaural_pkl_file):
        with open(binaural_pkl_file, 'rb') as file:
            audio = pickle.load(file)
    else:
        audio, input_sr = librosa.load(binaural_file, sr=target_sr, mono=False)
        with open(binaural_pkl_file, 'wb') as file:
            pickle.dump(audio, file)


    return rgb, audio


def read_near_audio(binaural_file, target_sr):
    binaural_pkl_file = binaural_file.replace('.wav', f'_{target_sr}.pkl')
    if os.path.exists(binaural_pkl_file):
        with open(binaural_pkl_file, 'rb') as file:
            audio = pickle.load(file)
    else:
        audio, input_sr = librosa.load(binaural_file, sr=target_sr, mono=False)
        with open(binaural_pkl_file, 'wb') as file:
            pickle.dump(audio, file)
    audio = audio.reshape(-1)
    audio = np.stack([audio, audio], axis=0)
    return audio


    
class ReplayNVASDataset(Dataset):
    def __init__(self, 
                 cfg,
                 data_root,
                 split='train',
                 scene=None,
                 gaussian=None
                 ):
        super().__init__()
        self.split = split
        self.cfg = cfg
        self.estimated_heads = None
        self.audio_len = self.sr = self.cfg.dataset.sr

        with open(os.path.join(data_root, 'v3/metadata_v2.json'), 'r') as fo:
            self.metadata = json.load(fo)
        scene_splits = SCENE_SPLITS['v3']
        ori_clip_dirs = [x for x in list(self.metadata.keys()) if x.split('/')[-2] in scene_splits[split]]
        
        train_frames_end = int(len(ori_clip_dirs) * 0.8)

        self.gaussians = GaussianModel(3)
        self.scene = Scene(cfg.dataset.data_root+f'/cam_imags/', cfg.dataset.data_root+f'/cam_imags/', self.gaussians, sh_degree=3, align_grids=None)


        self.train_cams = [2, 3, 4, 5, 6, 7, 8]
        self.test_cams = [2, 3, 4, 5, 6, 7, 8]

        self.pair_list = []
        self.clip_dirs = []
        self.cam_dict = {}

        cam_json_path = cfg.dataset.data_root+f'/cam_imags/gs_cameras.json'
        with open(cam_json_path, 'r') as file:
            cam_dict = json.load(file)
        valid_cams = []
        for cam in cam_dict:
            if cam['height'] < 100 and cam['width'] < 200:
                continue
            valid_cams.append(cam)

        
        for cam in valid_cams:
            cam_key = cam['img_name']
            video_name = cam_key.split('_')[0]
            cam_id = cam_key.split('_')[1]
            self.cam_dict[video_name+'/'+cam_id] = cam
            
        for clip_idx, clip_dir in enumerate(ori_clip_dirs):
            if clip_idx % 3 != 0:
                continue
            self.clip_dirs.append(clip_dir)


        if split is 'train':
            for clip_dir in self.clip_dirs:
                for cam in self.train_cams:
                    file = os.path.join(clip_dir, f'{cam}.png')
                    key = file.split('/')[-3] + '/' + f'{cam}'
                    if os.path.exists(file) and key in self.cam_dict:
                        self.pair_list.append((clip_dir, file))
        elif split is 'val':
            for clip_dir in self.clip_dirs:
                for cam in self.test_cams:
                    file = os.path.join(clip_dir, f'{cam}.png')
                    key = file.split('/')[-3] + '/' + f'{cam}'
                    if os.path.exists(file) and key in self.cam_dict:
                        self.pair_list.append((clip_dir, file))
        elif split is 'test':
            for idx, clip_dir in enumerate(self.clip_dirs):
                for cam in self.test_cams:
                    file = os.path.join(clip_dir, f'{cam}.png')
                    key = file.split('/')[-3] + '/' + f'{cam}'
                    if os.path.exists(file) and key in self.cam_dict:
                        self.pair_list.append((clip_dir, file))
        
        print(f'Number of clip is {len(self.clip_dirs)} for {self.split.upper()}')
        print(f'Number of samples is {len(self.pair_list)} for {self.split.upper()}')

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, index):
        clip_dir, file = self.pair_list[index]
        file_1, file_2 = file, file
        
        frame_idx = int(file_1.split('/')[-2])
        v2, a2 = read_clip(file_2, self.split, self.sr)
        # source
        near_file = os.path.join(clip_dir, 'near.wav')
        a0 = read_near_audio(near_file, self.sr)
        a1 = a0
        input_tgt_rgb = to_tensor(v2).permute(2, 0, 1) / 255.0
        # True in the ori paper, remove low band noise
        a1 = np.array(butter_bandpass_filter(a1, 150, self.sr // 2 - 1, self.sr, order=5).copy(), dtype=np.float32)
        a2 = np.array(butter_bandpass_filter(a2, 150, self.sr // 2 - 1, self.sr, order=5).copy(), dtype=np.float32)

        # True in the ori paper, remove delay
        delay = int(gcc_phat(np.mean(a2, axis=0), np.mean(a1, axis=0), self.sr,
                                    return_second=False)[0])
        delay = np.clip(delay, a_min=-200, a_max=200)
        if delay > 0:
            a1 = np.pad(a1, ((0, 0), (delay, 0)))[:, :-delay]
        else:
            a1 = np.pad(a1, ((0, 0), (0, -delay)))[:, -delay:]


        sample = dict()
        a1, a2 = self.process_audio(a1), self.process_audio(a2)

        if self.split == 'train' and a1.shape[0] > self.audio_len and a2.shape[0] > self.audio_len:
            random_start = np.random.randint(a1.shape[0] - self.audio_len)
            input_src_wav = to_tensor(a1[:, random_start: random_start + self.audio_len])
            input_tgt_wav = to_tensor(a2[:, random_start: random_start + self.audio_len])
        else:
            input_src_wav = to_tensor(a1[:, :self.audio_len])
            input_tgt_wav = to_tensor(a2[:, :self.audio_len])

        src_index = int(file_1.split('/')[-1][:-4])
        tgt_index = int(file_2.split('/')[-1][:-4])
        scene_index = file_1.split('/')[-3]
        
        src_pos = np.array(self.cam_dict[scene_index+'/'+str(src_index)]['position'])
        src_rot = np.array(self.cam_dict[scene_index+'/'+str(src_index)]['rotation'])
        tgt_pos = np.array(self.cam_dict[scene_index+'/'+str(tgt_index)]['position'])
        tgt_rot = np.array(self.cam_dict[scene_index+'/'+str(tgt_index)]['rotation'])
        
        fx, fy = self.cam_dict[scene_index+'/'+str(tgt_index)]['fx'], self.cam_dict[scene_index+'/'+str(tgt_index)]['fy']
        width, height = self.cam_dict[scene_index+'/'+str(tgt_index)]['width'], self.cam_dict[scene_index+'/'+str(tgt_index)]['height']
        cx = width / 2
        cy = height / 2

        intri = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]])
        cam_pose = torch.from_numpy(np.vstack([np.hstack([src_pos.reshape(-1), src_rot.reshape(-1)]), np.hstack([tgt_pos.reshape(-1), tgt_rot.reshape(-1)])]))[1] # 2, 12

        bboxes = np.load(file_2.replace('.png', '.npy'))
        spk_ids, counts = np.unique(bboxes[:, 4], return_counts=True)
        spk_idx = np.nonzero(bboxes[:, 4] == spk_ids[np.argmax(counts)])[0][0]
        
        input_bboxes = to_tensor(bboxes[spk_idx][:4] * np.array([width, height, width, height]))

        return int(scene_index.split('-')[1]), cam_pose.float(), input_tgt_wav.float(), input_src_wav.float(), input_bboxes, intri, tgt_index

            
    def process_audio(self, audio):
        if audio.shape[1] < self.audio_len:
            audio = np.pad(audio, ((0, 0), (0, self.audio_len - audio.shape[1])))
        return audio


def build_dataset(cfg):
    train_dataset = ReplayNVASDataset(cfg, cfg.dataset.data_root, split='train')
    eval_dataset = ReplayNVASDataset(cfg, cfg.dataset.data_root, split='val', scene=train_dataset.scene, gaussian=train_dataset.gaussians)
    return train_dataset, eval_dataset

def build_single_dataset(cfg, split):
    single_dataset = ReplayNVASDataset(cfg, cfg.dataset.data_root, split=split)
    return single_dataset
