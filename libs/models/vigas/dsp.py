# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE-3RD-PARTY-NVAS file in the root directory of this source tree.



import librosa
import numpy as np
import sofa
import soundfile as sf
import speechmetrics
import torch
import torch as th
import torchaudio as ta
from scipy.constants import speed_of_sound
from scipy.signal import butter, correlate, lfilter, sosfilt, sosfreqz
from torch import nn


class DSP(torch.nn.Module):
    def __init__(self, args):
        super(DSP, self).__init__(args)
        self.args = args
        self.learnable = False

        self.hrtf = Hrtf()

        self.first_val = True
        self.metrics = ['stft_distance', 'mag_distance', 'delay', 'lr_ratio', 'lr_ratio_peak', 'l2_distance',
                        'rt60_error']
        self.speech_metric_names = {'stoi', 'sisdr', 'bsseval', 'mosnet'}.intersection(self.metrics)
        self.speech_metrics = speechmetrics.load(self.speech_metric_names, window=None)

    def audio_synthesis(self, batch, batch_idx, phase=None):
        for key, value in batch.items():
            batch[key] = value.to(device=self.device, dtype=torch.float)

        pred_wav = []
        for i in range(batch['src_wav'].shape[0]):
            if self.args.dataset == 'synthetic':
                speaker_pose = batch['speaker_pose'][i].cpu().numpy()
                speaker_pose_wrt_tgt = batch['speaker_pose_wrt_tgt'][i].cpu().numpy()
                mono_wav = self.hrtf.apply_inv_hrtf(batch['src_wav'][i].cpu() * self.args.base_gain,
                                                    azimuth=np.rad2deg(speaker_pose[1]), elevation=0, distance=speaker_pose[0])
                pred_wav.append(self.hrtf.apply_hrtf(mono_wav, azimuth=np.rad2deg(speaker_pose_wrt_tgt[1]),
                                elevation=0, distance=speaker_pose_wrt_tgt[0]))
            else:
                mono_wav = self.hrtf.apply_inv_hrtf(batch['src_wav'][i].cpu() * self.args.base_gain,
                                                    azimuth=np.rad2deg(batch['src_azimuth'][i].cpu().numpy()),
                                                    elevation=np.rad2deg(batch['src_elevation'][i].cpu().numpy()),
                                                    distance=batch['src_distance'][i].cpu().numpy())
                pred_wav.append(self.hrtf.apply_hrtf(mono_wav,
                                                     azimuth=np.rad2deg(batch['tgt_azimuth'][i].cpu().numpy()),
                                                     elevation=np.rad2deg(batch['tgt_elevation'][i].cpu().numpy()),
                                                     distance=batch['tgt_distance'][i].cpu().numpy()))
        pred_wav = torch.stack(pred_wav, dim=0).to(self.device)[:, :, :self.args.audio_len]

        return {'pred': pred_wav, 'tgt': batch['tgt_wav'], 'batch': batch}


class Hrtf:
    def __init__(
        self,
        hrtf_file: str = "data/Kemar_HRTF_sofa.sofa"
    ):
        hrtf = sofa.Database.open(hrtf_file)
        pos = hrtf.Source.Position.get_values().astype(np.int64)
        self.az, self.el = pos[:, 0], pos[:, 1]
        # filters
        fltrs = hrtf.Data.IR.get_values().astype(np.float32)
        self.filters = {(int(self.az[i]), int(self.el[i])): th.from_numpy(fltrs[i]) for i in range(fltrs.shape[0])}
        # inverse filters
        self.inv_filters = {k: self._invert_fltr(v) for k, v in self.filters.items()}

    def _invert_fltr(self, h):
        H = th.fft.fft(h)
        H_inv = th.conj(H) / (th.abs(H) ** 2 + 1e-4)  # Wiener filter, 1e-4 is a noise estimate to compensate for low energy elements
        h_inv = th.real(th.fft.ifft(H_inv))
        return h_inv

    def _get_fltr_idx(self, azimuth, elevation):
        azimuth = self.az[np.abs(self.az - azimuth).argmin()]
        elevation = self.el[np.abs(self.el - elevation).argmin()]
        return (azimuth, elevation)

    def apply_hrtf(self, mono_signal: th.Tensor, azimuth: float, elevation: float, distance: float, base_gain: float = 1.0):
        """
        mono_signal: 1 x T tensor
        azimuth: the azimuth in degrees between 0 and 360
        elevation: the elevation in degrees between -90 and 90
        distance: the distance in meters
        returns a 2 x T tensor containing the binauralized signal
        """
        # binauralize
        h = self.filters[self._get_fltr_idx(azimuth, elevation)]
        h = th.flip(h, dims=(-1,))
        mono_signal = th.cat([mono_signal, mono_signal], dim=0).unsqueeze(0)  # duplicate signal and add batch dimension
        mono_signal = th.nn.functional.pad(mono_signal, pad=(h.shape[-1], 0))
        binaural = th.nn.functional.conv1d(mono_signal, h.unsqueeze(1).to(mono_signal.device), groups=2).squeeze(0)
        # adjust gain based on distance (HRTF is measured at 2m distance but gain information is lost due to normalization)
        ref_dist = 0.2
        dist = max(ref_dist, distance)
        gain = base_gain * ref_dist / dist
        return binaural * gain

    def apply_inv_hrtf(self, binaural_signal: th.Tensor, azimuth: float, elevation: float, distance: float, base_gain: float = 1.0):
        """
        binaural_signal: 2 x T tensor
        azimuth: the azimuth in degrees between 0 and 360
        elevation: the elevation in degrees between -90 and 90
        distance: the distance in meters
        returns a 1 x T tensor containing the mono signal as the mixture of left ear and right ear inverse transformation
        """
        h = self.inv_filters[self._get_fltr_idx(azimuth, elevation)]
        h = th.flip(h, dims=(-1,))
        binaural_signal = binaural_signal.unsqueeze(0)  # add batch dimension
        binaural_signal = th.nn.functional.pad(binaural_signal, pad=(h.shape[-1], 0))
        mono = th.nn.functional.conv1d(binaural_signal, h.unsqueeze(1), groups=2).squeeze(0)
        mono = th.mean(mono, dim=0, keepdim=True)
        # adjust gain based on distance (HRTF is measured at 2m distance but gain information is lost due to normalization)
        ref_dist = 0.2
        dist = max(ref_dist, distance)
        gain = base_gain * ref_dist / dist
        return mono / gain

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def compute_panning(left_channel, right_channel, fs, lowcut=20, highcut=20000):
    # Filter both channels
    left_filtered = bandpass_filter(left_channel, lowcut, highcut, fs)
    right_filtered = bandpass_filter(right_channel, lowcut, highcut, fs)
    
    # Compute power of the filtered signals
    power_left = (left_filtered**2).sum(-1)
    power_right = (right_filtered**2).sum(-1)
    total_power = power_left + power_right
    
    # Avoid division by zero
    if total_power == 0:
        return 0  # This could be handled differently based on your application
    
    # Compute panning
    panning = (power_right - power_left) / total_power
    return panning

def find_angle(gt_signal, hrtf, sr, low=5500, high=20000):
    mono_signal = gt_signal.mean(0)[None, ...]
    gt_panning = compute_panning(gt_signal[0], gt_signal[1], sr, low, high)
    min_panning_diff = 100000
    best_azimuth = None
    best_elevation = None

    binaural_signal = hrtf.apply_hrtf(mono_signal, azimuth=0, elevation=0, distance=1.0)
    gt_energy = (gt_signal**2).sum(-1).mean().sqrt()
    cur_energy = (binaural_signal**2).sum(-1).mean().sqrt()
    distance = cur_energy / gt_energy
    distance = 0.5
    # distance = 2

    for azimuth in range(-90, 91, 10):
        cur_azimuth = (360 + azimuth) % 360
        binaural_signal = hrtf.apply_hrtf(mono_signal, azimuth=cur_azimuth, elevation=0, distance=distance)
        cur_panning = compute_panning(binaural_signal[0], binaural_signal[1], sr, 5500, 20000)
        cur_energy = (binaural_signal**2).sum(-1).mean().sqrt()
        if abs(cur_panning-gt_panning) < min_panning_diff:
            best_azimuth = cur_azimuth
            min_panning_diff = abs(cur_panning-gt_panning)
    
    min_panning_diff = 100000
    best_azimuth_1 = None
    for azimuth in range(-10+best_azimuth, best_azimuth+10):
        cur_azimuth = (360 + azimuth) % 360
        binaural_signal = hrtf.apply_hrtf(mono_signal, azimuth=cur_azimuth, elevation=0, distance=distance)
        cur_panning = compute_panning(binaural_signal[0], binaural_signal[1], sr, 5500, 20000)
        cur_energy = (binaural_signal**2).sum(-1).mean().sqrt()
        energy_diff = abs(cur_energy - gt_energy)
        if abs(cur_panning-gt_panning) < min_panning_diff:
            best_azimuth_1 = cur_azimuth
            min_panning_diff = abs(cur_panning-gt_panning)
            
    final_azimuth = best_azimuth_1
    # print(min_panning_diff, gt_panning)
    return final_azimuth
    
    # min_panning_diff = 100000
    # for elevation in range(-90, 91, 10):
    #     binaural_signal = hrtf.apply_hrtf(mono_signal, azimuth=final_azimuth, elevation=elevation, distance=distance)
    #     cur_panning = compute_panning(binaural_signal[0], binaural_signal[1], sr, 5500, 20000)
    #     cur_energy = (binaural_signal**2).sum(-1).mean().sqrt()
    #     energy_diff = abs(cur_energy - gt_energy)
    #     # print(elevation, cur_energy, gt_energy)
    #     if abs(cur_panning-gt_panning) + energy_diff < min_panning_diff:
    #         best_elevation = elevation
    #         min_panning_diff = abs(cur_panning-gt_panning) + energy_diff
    
    # min_panning_diff = 100000
    # best_elevation_1 = None
    # for elevation in range(best_elevation-10, best_elevation+10):
    #     binaural_signal = hrtf.apply_hrtf(mono_signal, azimuth=final_azimuth, elevation=elevation, distance=distance)
    #     cur_panning = compute_panning(binaural_signal[0], binaural_signal[1], sr, 5500, 20000)
    #     cur_energy = (binaural_signal**2).sum(-1).mean().sqrt()
    #     energy_diff = abs(cur_energy - gt_energy)
    #     # print(elevation, cur_energy, gt_energy)
    #     if abs(cur_panning-gt_panning) + energy_diff < min_panning_diff:
    #         best_elevation_1 = elevation
    #         min_panning_diff = abs(cur_panning-gt_panning) + energy_diff

    # final_elevation = best_elevation_1
    # print(final_azimuth, final_elevation, min_panning_diff)
    # import pdb; pdb.set_trace()
    # binaural_signal = hrtf.apply_hrtf(mono_signal, azimuth=final_azimuth, elevation=final_elevation, distance=distance)
    # cur_panning = compute_panning(binaural_signal[0], binaural_signal[1], sr, 5500, 20000)
    # cur_energy = (binaural_signal**2).sum(-1).mean().sqrt()
    # energy_diff = abs(cur_energy - gt_energy)


    # return final_azimuth, final_elevation



if __name__ == '__main__':
    ### EXAMPLE ###
    hrtf = Hrtf()
    gt_signal, sr = ta.load("31.wav")
    import pdb; pdb.set_trace()
    azimuth = find_angle(gt_signal, hrtf, sr, low=5500, high=20000)
    # gt_signal1, sr = ta.load("/mnt/data1/SpatialAudioGen/data/scenes/trevi_fountain/collect_clean_audios/AF1QipNBRgcNLE0C-G3Am98qywagZZ3J-fO-5Y0UhLfI_frame_0022.wav")
    mono_signal = gt_signal.mean(0)[None, ...]
    

    # for i in range(1, 24):
    #     audio_path = f'/mnt/data1/SpatialAudioGen/data/scenes/trevi_fountain/collect_clean_audios/AF1QipNBRgcNLE0C-G3Am98qywagZZ3J-fO-5Y0UhLfI_frame_{str(i).zfill(4)}.wav'
    #     gt_signal, sr = ta.load(audio_path)
    #     azimuth = find_angle(gt_signal, hrtf)
    #     print(i, azimuth)
    binaural_signal = hrtf.apply_hrtf(mono_signal, azimuth=360-5, elevation=0, distance=1.0)
#     print(compute_panning(binaural_signal[0], binaural_signal[1], sr, 5500, 20000))

#     binaural_signal1 = hrtf.apply_hrtf(mono_signal, azimuth=360-10, elevation=0, distance=1.0)
#     print(compute_panning(binaural_signal1[0], binaural_signal1[1], sr, 5500, 20000))

#     binaural_signal2 = hrtf.apply_hrtf(mono_signal, azimuth=0, elevation=0, distance=1.0)
#     binaural_signal3 = hrtf.apply_hrtf(mono_signal, azimuth=5, elevation=0, distance=1.0)
#     binaural_signal4 = hrtf.apply_hrtf(mono_signal, azimuth=360-10, elevation=0, distance=1.0)
#     binaural_signal5 = hrtf.apply_hrtf(mono_signal, azimuth=10, elevation=0, distance=1.0)
#     print(compute_panning(gt_signal[0], gt_signal[1], sr, 5500, 20000))
#     print(compute_panning(gt_signal1[0], gt_signal1[1], sr, 5500, 20000))
   
#     print(compute_panning(binaural_signal2[0], binaural_signal2[1], sr, 5500, 20000))
#     print(compute_panning(binaural_signal3[0], binaural_signal3[1], sr, 5500, 20000))
#     print(compute_panning(binaural_signal4[0], binaural_signal4[1], sr, 5500, 20000))
#     print(compute_panning(binaural_signal5[0], binaural_signal5[1], sr, 5500, 20000))
#     import pdb; pdb.set_trace()
#     # mono_recon = hrtf.apply_inv_hrtf(binaural_signal, azimuth=-10, elevation=0, distance=1.0)
    ta.save("binaural.wav", binaural_signal, sr)
#     ta.save("binaural_2.wav", binaural_signal1, sr)
#     # ta.save("mono_reconstruction.wav", mono_recon, sr)

# # python libs/models/vigas/dsp.py
