"""
Code of "AV-Cloud: Spatial Audio Rendering Through Audio-Visual Cloud Splatting" 

Copyright (c) 2023-2024 University of Washington. 

Developed in UW NeuroAI Lab by Mingfei Chen (lasiafly@uw.edu).
"""



import os
import pickle

import audioldm_eval.audio as Audio
import cdpam
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from audioldm_eval import (calculate_fid, calculate_isc, calculate_kid,
                           calculate_kl)
from audioldm_eval.audio.tools import (load_json, load_pickle, save_pickle,
                                       write_json)
from audioldm_eval.datasets.load_mel import (MelPairedDataset, WaveDataset,
                                             load_npy_data)
from audioldm_eval.feature_extractors.panns import Cnn14
from scipy.signal import hilbert
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from ssr_eval.metrics import AudioMetrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from libs.models.vigas.dsp import bandpass_filter


def convert_float32(obj):
    if isinstance(obj, dict):
        for key in obj:
            obj[key] = convert_float32(obj[key])
    elif isinstance(obj, list):
        obj = [convert_float32(item) for item in obj]
    elif isinstance(obj, np.float32):
        return float(obj)
    return obj


def STFT_L2_distance(predicted_binaural, gt_binaural):
    #channel1
    predicted_spect_channel1 = librosa.core.stft(np.asfortranarray(predicted_binaural[0,:]), n_fft=512, hop_length=160, win_length=400, center=True)
    gt_spect_channel1 = librosa.core.stft(np.asfortranarray(gt_binaural[0,:]), n_fft=512, hop_length=160, win_length=400, center=True)
    real = np.expand_dims(np.real(predicted_spect_channel1), axis=0)
    imag = np.expand_dims(np.imag(predicted_spect_channel1), axis=0)
    predicted_realimag_channel1 = np.concatenate((real, imag), axis=0)
    real = np.expand_dims(np.real(gt_spect_channel1), axis=0)
    imag = np.expand_dims(np.imag(gt_spect_channel1), axis=0)
    gt_realimag_channel1 = np.concatenate((real, imag), axis=0)
    channel1_distance = np.mean(np.power((predicted_realimag_channel1 - gt_realimag_channel1), 2))

    #channel2
    predicted_spect_channel2 = librosa.core.stft(np.asfortranarray(predicted_binaural[1,:]), n_fft=512, hop_length=160, win_length=400, center=True)
    gt_spect_channel2 = librosa.core.stft(np.asfortranarray(gt_binaural[1,:]), n_fft=512, hop_length=160, win_length=400, center=True)
    real = np.expand_dims(np.real(predicted_spect_channel2), axis=0)
    imag = np.expand_dims(np.imag(predicted_spect_channel2), axis=0)
    predicted_realimag_channel2 = np.concatenate((real, imag), axis=0)
    real = np.expand_dims(np.real(gt_spect_channel2), axis=0)
    imag = np.expand_dims(np.imag(gt_spect_channel2), axis=0)
    gt_realimag_channel2 = np.concatenate((real, imag), axis=0)
    channel2_distance = np.mean(np.power((predicted_realimag_channel2 - gt_realimag_channel2), 2))

    #sum the distance between two channels
    stft_l2_distance = channel1_distance + channel2_distance
    return float(stft_l2_distance)

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
    spec = torch.stft(wav, n_fft=512, hop_length=160, win_length=400,
                      window=torch.hamming_window(400, device=wav.device), pad_mode='constant',
                      return_complex=False)

    return spec


def eval_mag(wav, log=False):
    stft = torch.stft(wav, n_fft=512, hop_length=160, win_length=400,
                      window=torch.hamming_window(400, device=wav.device), pad_mode='constant',
                      return_complex=True)
    if log:
        mag = torch.log(stft.abs() + 1e-5)
    else:
        mag = stft.abs()

    return mag


class Evaluator:
    def __init__(self, cfg, seq_name, sampling_rate=32000, device='cuda', backbone="cnn14"):
        self.cfg = cfg
        self.seq_name = seq_name
        self.device = device
        self.backbone = backbone
        self.sampling_rate = sampling_rate

        self.metrics = {
            'median_scale': [],
            'STFT_L2': [],
            'left_right_err': [],
            'env': [],
            'dpam': [],
            'sdr': [],
            'stft_mse': [],
            'mag_mse': [],
            'log_mag_mse': [],
        }
        self.fig = plt.figure()
        self.figplot = self.fig.add_subplot(2, 1, 1)
        self.figplot1 = self.fig.add_subplot(2, 1, 2)

    def measure_sdr(self, true_in, gen_in):
        num = np.sum(true_in**2)
        den = np.sum((true_in - gen_in)**2)
        scores = 10*np.log10(num/den)
        return scores
    

    def calculate_psnr_ssim(self, pairedloader, same_name=True):
        if same_name == False:
            return {"psnr": -1, "ssim": -1}
        psnr_avg = []
        ssim_avg = []
        for mel_gen, mel_target, filename, _ in (pairedloader):
            mel_gen = mel_gen.cpu().numpy()[0]
            mel_target = mel_target.cpu().numpy()[0]
            psnrval = psnr(mel_gen, mel_target)
            if np.isinf(psnrval):
                print("Infinite value encountered in psnr %s " % filename)
                continue
            psnr_avg.append(psnrval)
            data_range = max(np.max(mel_gen), np.max(mel_target)) - min(np.min(mel_gen), np.min(mel_target))
            ssim_avg.append(ssim(mel_gen, mel_target, data_range=data_range))
        return {"psnr": np.mean(psnr_avg), "ssim": np.mean(ssim_avg)}


    def evaluate(self, pred_wav, gt_waveform, sr=22050, limit_num=None, recalculate=True, same_name=True):
        self.metrics['median_scale'].append((np.median(abs(pred_wav))/np.median(abs(gt_waveform))))
        self.metrics['STFT_L2'].append(STFT_L2_distance(pred_wav, gt_waveform))
        gt_waveform = gt_waveform[..., :pred_wav.shape[-1]]
        self.metrics['env'].append(Envelope_distance(pred_wav, gt_waveform))
        gt_waveform_ori = gt_waveform
        gt_energy = 10 * np.log10((gt_waveform**2).sum(-1)+ 1e-5)
        pred_energy = 10 * np.log10((pred_wav**2).sum(-1)+ 1e-5)

        pred_lr_ratio = 10 * np.log10(((pred_wav[0]**2).sum(-1)+ 1e-5) / ((pred_wav[1]**2).sum(-1) + 1e-5))
        tgt_lr_ratio = 10 * np.log10(((gt_waveform[0]**2).sum(-1)+ 1e-5) / ((gt_waveform[1]**2).sum(-1) + 1e-5))
        self.metrics['left_right_err'].append(abs(pred_lr_ratio - tgt_lr_ratio))

        loss_fn = cdpam.CDPAM()
        generate_files_path = f'work_dirs/{self.cfg.output_dir}/tmp_pred'
        groundtruth_path = f'work_dirs/{self.cfg.output_dir}/tmp_gt'
        if not os.path.exists(generate_files_path):
            os.makedirs(generate_files_path)
            os.makedirs(groundtruth_path)
        sf.write(generate_files_path+'/1.wav', pred_wav.T, sr)
        sf.write(groundtruth_path+'/1.wav', gt_waveform.T, sr)
        wav_ref = cdpam.load_audio(groundtruth_path+'/1.wav')
        wav_out = cdpam.load_audio(generate_files_path+'/1.wav')
        self.metrics['dpam'].append(loss_fn.forward(wav_ref,wav_out).data.cpu().numpy()[0])
        sdr = self.measure_sdr(gt_waveform, pred_wav)
        self.metrics['sdr'].append(sdr)

        pred_wav_tensor = torch.from_numpy(pred_wav)
        gt_waveform_tensor = torch.from_numpy(gt_waveform)
        pred_spec_tensor, tgt_spec_tensor = eval_stft(pred_wav_tensor), eval_stft(gt_waveform_tensor)
        self.metrics['stft_mse'].append(F.mse_loss(pred_spec_tensor, tgt_spec_tensor).data.cpu().numpy().mean())
        pred_spec_tensor, tgt_spec_tensor = eval_mag(pred_wav_tensor), eval_mag(gt_waveform_tensor)
        self.metrics['mag_mse'].append(F.mse_loss(pred_spec_tensor, tgt_spec_tensor).data.cpu().numpy().mean())
        pred_spec_tensor, tgt_spec_tensor = eval_mag(pred_wav_tensor, log=True), eval_mag(gt_waveform_tensor, log=True)
        self.metrics['log_mag_mse'].append(F.mse_loss(pred_spec_tensor, tgt_spec_tensor).data.cpu().numpy().mean())

        torch.manual_seed(0)

        num_workers = 6

        out = {}



    def summarize(self):
        result_path = os.path.join(self.cfg.result_dir,
            self.seq_name, 'metrics.pkl')
        os.system('mkdir -p {}'.format(os.path.dirname(result_path)))
        metrics = {}
        for key in self.metrics:
            metrics[key] = np.mean(self.metrics[key]),
        with open(result_path, 'wb') as file:
            pickle.dump(metrics, file)
        return metrics
    