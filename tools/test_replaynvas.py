"""
Code of "AV-Cloud: Spatial Audio Rendering Through Audio-Visual Cloud Splatting" 

Copyright (c) 2023-2024 University of Washington. 

Developed in UW NeuroAI Lab by Mingfei Chen (lasiafly@uw.edu).
"""



from __future__ import division, print_function, with_statement

import argparse
import os
import pickle
import random
import time
from importlib import import_module as impm

import _init_paths
import einops
import librosa
import numpy as np
import torch
from scipy.signal import hilbert
from tqdm import tqdm

from configs import cfg, update_config
from libs.evaluators.gen_eval import Evaluator
from libs.models.vigas.visual_net import VisualNet
from libs.utils import misc


def load_rt60_estimator(device):
    RT60_ESTIMATOR = VisualNet(use_rgb=False, use_depth=False, use_audio=True)
    pretrained_weights = 'data/avcloud_data/models/rt60_estimator.pth'
    RT60_ESTIMATOR.load_state_dict(torch.load(pretrained_weights, map_location='cpu')['predictor'])
    RT60_ESTIMATOR.to(device=device).eval()

    return RT60_ESTIMATOR


def estimate_rt60(estimator, wav):
    stft = torch.stft(wav, n_fft=512, hop_length=160, win_length=400, window=torch.hamming_window(400, device=wav.device),
                      pad_mode='constant', return_complex=True)
    spec = torch.log1p(stft.abs()).unsqueeze(1)
    with torch.no_grad():
        estimated_rt60 = estimator(spec.float())
    return estimated_rt60


def Envelope_distance(predicted_binaural, gt_binaural):
    #channel1
    pred_env_channel1 = np.abs(hilbert(predicted_binaural[0,:]))
    gt_env_channel1 = np.abs(hilbert(gt_binaural[0,:]))
    channel1_distance = np.sqrt(np.mean((gt_env_channel1 - pred_env_channel1)**2))

    #channel2
    pred_env_channel2 = np.abs(hilbert(predicted_binaural[1,:]))
    gt_env_channel2 = np.abs(hilbert(gt_binaural[1,:]))
    channel2_distance = np.sqrt(np.mean((gt_env_channel2 - pred_env_channel2)**2))

    #sum the distance between two channels
    envelope_distance = channel1_distance + channel2_distance
    return float(envelope_distance)


def eval_stft(wav):
    # return real and imaginary components as two channels
    assert len(wav.shape) == 2
    spec = torch.stft(wav, n_fft=512, hop_length=160, win_length=400,
                      window=torch.hamming_window(400, device=wav.device), pad_mode='constant',
                      return_complex=False)

    return spec


def eval_mag(wav, log=False):
    assert len(wav.shape) == 2
    stft = torch.stft(wav, n_fft=512, hop_length=160, win_length=400,
                      window=torch.hamming_window(400, device=wav.device), pad_mode='constant',
                      return_complex=True)
    if log:
        mag = torch.log(stft.abs() + 1e-5)
    else:
        mag = stft.abs()

    return mag



def parse_args():
    parser = argparse.ArgumentParser(description='Neural Acoustic')
    parser.add_argument(
        '--cfg',
        dest='yaml_file',
        help='experiment configure file name, e.g. configs/base_config.yaml',
        required=True,
        type=str)
    # default distributed training
    parser.add_argument(
        '--distributed',
        action='store_true',
        default=False,
        help='if use distribute train')
    parser.add_argument(
        '--dist-url',
        dest='dist_url',
        default='tcp://10.5.38.36:23456',
        type=str,
        help='url used to set up distributed training')
    parser.add_argument(
        '--world-size',
        dest='world_size',
        default=1,
        type=int,
        help='number of nodes for distributed training')
    parser.add_argument(
        '--rank',
        default=0,
        type=int,
        help='node rank for distributed training, machine level')

    parser.add_argument(
        'opts',
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    return args


args = parse_args()

update_config(cfg, args)
ngpus_per_node = torch.cuda.device_count()

# torch seed
seed = cfg.seed + misc.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


criterion = getattr(impm(cfg.train.criterion_file), 'Criterion')(cfg)
if cfg.device == 'cuda':
    torch.cuda.set_device(0)
device = torch.device(cfg.device)

split = 'test'
test_dataset = getattr(impm(cfg.dataset.name), 'build_single_dataset')(cfg, split)

model = getattr(impm(cfg.model.file), 'build_model')(cfg, test_dataset.gaussians, test_dataset.scene)
model = torch.nn.DataParallel(model).to(device)

model_without_ddp = model.module
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(n_parameters)


test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    drop_last=cfg.dataset.test.drop_last,
    num_workers=cfg.workers,
    sampler=None
)
evaluator = Evaluator(cfg, cfg.model.file)

model.eval()
criterion.eval()
times = []

###############
resume_path = cfg.model.resume_path
################
if os.path.exists(resume_path):
    checkpoint = torch.load(resume_path, map_location='cpu')
    # resume
    if 'state_dict' in checkpoint:
        model.module.load_state_dict(checkpoint['state_dict'], strict=True)
        print(f'==> model loaded from {resume_path} \n')


def _read_inputs(batch, device):
    for k in range(len(batch)):
        if isinstance(batch[k], tuple) or isinstance(batch[k], list):
            batch[k] = [b.to(device) for b in batch[k]]
        if isinstance(batch[k], dict):
            batch[k] = {key: value.to(device) for key, value in batch[k].items()}
        else:
            batch[k] = batch[k].to(device)
    return batch


count = 0
stats = {
    'rte': [],
    'mag': [],
    'lre': [],
    'env': [],
}


with torch.no_grad():
    for ori_val_data in tqdm(test_loader):
        val_data = _read_inputs(ori_val_data, device)
        index, cam_pose, gt_waveform, source_gt_audio, input_bboxes, intri, data_index = val_data
        B = len(gt_waveform)
        start = time.time()
        pred_wav = model(cam_pose, source_gt_audio, input_bboxes, intri, is_val=True)
        torch.cuda.synchronize()
        end = time.time()
        elapsed = end - start
        times.append(elapsed)
        
        ### evaluate
        gt_waveform = gt_waveform[..., :pred_wav.shape[-1]]
        np_gt_ir = gt_waveform.reshape(2, -1).data.cpu().numpy()
        np_pred_ir = pred_wav.reshape(2, -1).data.cpu().numpy()
        pred_wav = pred_wav.reshape(B, 2, -1)
        tgt_wav = gt_waveform.reshape(B, 2, -1)

        if tgt_wav[:, 0].pow(2).sum(-1) == 0:
            continue

        evaluator.evaluate(np_pred_ir, np_gt_ir, sr=cfg.dataset.sr)
        
        stats['env'].append(Envelope_distance(np_pred_ir, np_gt_ir[..., :np_pred_ir.shape[-1]]))

        rt60_estimator = load_rt60_estimator(device)
        pred_rt60 = estimate_rt60(rt60_estimator, pred_wav.reshape(-1, pred_wav.shape[-1]))
        tgt_rt60 = estimate_rt60(rt60_estimator, tgt_wav[..., :pred_wav.shape[-1]].reshape(-1, pred_wav.shape[-1]))
        stats['rte'].append((pred_rt60 - tgt_rt60).abs().mean().item())


        pred_spec_l, tgt_spec_l = eval_mag(pred_wav[:, 0]), eval_mag(tgt_wav[:, 0])
        pred_spec_r, tgt_spec_r = eval_mag(pred_wav[:, 1]), eval_mag(tgt_wav[:, 1])
        stats['mag'].append(((pred_spec_l - tgt_spec_l).pow(2).sqrt().mean((1, 2)) + \
                                (pred_spec_r - tgt_spec_r).pow(2).sqrt().mean((1, 2))).mean().item())
        
        gt_energy = (tgt_wav[..., :pred_wav.shape[-1]]**2).sum(-1).reshape(2, -1)
        pred_energy = (pred_wav**2).sum(-1).reshape(2, -1)
        pred_lr_ratio = 10 * torch.log10((pred_wav[:, 0].pow(2).sum(-1)+ 1e-5) / (pred_wav[:, 1].pow(2).sum(-1) + 1e-5))
        tgt_lr_ratio = 10 * torch.log10((tgt_wav[:, 0].pow(2).sum(-1)+ 1e-5) / (tgt_wav[:, 1].pow(2).sum(-1) + 1e-5))
        stats['lre'].append(((pred_lr_ratio - tgt_lr_ratio).abs()).mean().item())

        count += 1


metrics = evaluator.summarize()
save_metrics = {}
save_metrics['infer_time'] = np.mean(times)
print('infer time in seconds: ', save_metrics['infer_time'])


save_metrics["dpam"] = metrics["dpam"][0]
print(f'dpam: ', save_metrics["dpam"])

for key in stats.keys():
    print(f'{key}: ', np.mean(stats[key]))
    save_metrics[key] = np.mean(stats[key])


os.makedirs("av_results", exist_ok=True)
with open(f'av_results/{cfg.output_dir}.pkl', 'wb') as file:
    pickle.dump(save_metrics, file)
