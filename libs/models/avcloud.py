"""
Code of "AV-Cloud: Spatial Audio Rendering Through Audio-Visual Cloud Splatting" 

Copyright (c) 2023-2024 University of Washington. 

Developed in UW NeuroAI Lab by Mingfei Chen (lasiafly@uw.edu).
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from libs.models.networks.batchnorm import SynchronizedBatchNorm2d
from libs.models.networks.encoder import (PositionalEncoder,
                                          embedding_module_log)
from libs.models.networks.mlp import basic_project2
from libs.models.networks.transformer import *
from libs.models.vigas.hyperconv import HyperConv
from libs.utils.sh_utils import eval_sh


class AVCloud(nn.Module):
    """Baseline AVCloud
    """

    def __init__(self, cfg, gaussian_model, scene, intermediate_ch = 128, time_num = 174,
        freq_num = 257, dilation=1):

        super(AVCloud, self).__init__()
        self.joint_emb_dim = cfg.model.joint_emb_dim
        self.gaussian_model = gaussian_model
        self.scene = scene
        self.model_type = cfg.model.model_type
        self.render_type = cfg.model.render_type

        N = self.gaussian_model.get_xyz.shape[0]
        
        self.freq_num = freq_num
        self.time_num = time_num
        self.dilation = dilation

        self._xyz = self.gaussian_model.get_xyz.detach()
        self._rgb_features = self.gaussian_model.get_features.contiguous().transpose(1, 2).detach()

        self._audio_features_f = nn.Parameter((torch.ones(N, intermediate_ch, 1)*0.25).requires_grad_(True))
        self.d_model = intermediate_ch
        

        self.pcd_av_proj = nn.Sequential(
            basic_project2(150, intermediate_ch),
            nn.PReLU(),
            basic_project2(intermediate_ch, self.d_model)
        )
        
        self.vec_embedder = embedding_module_log(num_freqs=10, ch_dim=1)
        self.vec_proj = nn.Sequential(
            basic_project2(63, intermediate_ch),
            nn.PReLU(),
            basic_project2(intermediate_ch, self.d_model),
        )

        self.diff_prj = nn.Linear(self.d_model, 1)
        self.mix_prj = nn.Linear(self.d_model, 1)

        if cfg.dataset.video == '_6':
            self.max_norm = 23.6032 # 6
        elif cfg.dataset.video == '_12' or cfg.dataset.video == '_13':
            self.max_norm = 183.2170 # 12
        else:
            self.max_norm = 25

        if self.model_type == 'full' or self.model_type == "sh":
            norm_layer = SynchronizedBatchNorm2d
            normalize_before=False
            decoder_layer = TransformerDecoderLayer(self.d_model, nhead=4, dim_feedforward=512,
                                                    dropout=0.1, activation="relu", normalize_before=normalize_before)
            decoder_norm = nn.LayerNorm(self.d_model)
            self.decoder = TransformerDecoder(decoder_layer, 3, decoder_norm,
                                            return_intermediate=False)

            self.freq_embedder = embedding_module_log(num_freqs=10, ch_dim=1)
            self.query_prj = nn.Sequential(nn.Linear(21, self.d_model), nn.ReLU(inplace=True))

        if self.render_type == "simple":
            if self.model_type == "full":
                self.kernel_size = 3
                self.post_process = HyperConv(input_size=self.d_model, ch_in=self.time_num, ch_out=self.time_num, kernel_size=self.kernel_size, dilation=self.dilation)
            else:
                self.kernel_size = 1
                self.time_freq_weights = nn.Parameter((torch.randn(N, self.freq_num, 4)).requires_grad_(True))
                self._acoustic_features_t = nn.Parameter((torch.randn(N, self.time_num*(self.kernel_size+1))).requires_grad_(True))
                self.time_weights = nn.Parameter((torch.randn(self.freq_num, self.time_num)).requires_grad_(True))
                if self.model_type == "simple-sh":
                    self.freq_weights = nn.Parameter((torch.randn(self.freq_num, self.d_model)).requires_grad_(True))
                
            self.out = nn.Sequential(nn.Conv2d(4, 16, 7, 1, 3),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(16, 16, 3, 1, 1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(16, 1, 3, 1, 1))
        else:
            import pdb; pdb.set_trace()

    def forward(self, cam_pose, source_gt_audio, is_val=False):
        """
        Args:
            cam_pose: (B, 12). Camera poses cam center: cam_pose[:, :3], rotation: cam_pose[:, 3:].reshape(3, 3)
            source_gt_audio: (B, 2, T). Source audio input waveform. The model will convert it to be the spatial audio that matches with GT.
        Return:
            out_pred_wav (B, 2, T). Predict binaural audio at the target viewpoint.
        """
        device = cam_pose.device
        B = len(cam_pose)
        hop_length = 127
        n_fft = 512
        window_length = 512
        torch_window = torch.hann_window(window_length=window_length).to(device)
        ref_spec_complex = torch.stft(source_gt_audio.detach().mean(1), n_fft=n_fft, hop_length=hop_length, win_length=window_length, center=True, window=torch_window, return_complex=True)
        ori_ref_mag = ref_mag = torch.abs(ref_spec_complex)
        left_phase = right_phase = torch.angle(ref_spec_complex)
        
        # Audio-Visual Anchors
        N_points = self.gaussian_model.get_xyz.shape[0]
        shs_view = self._rgb_features.transpose(1, 2)[:, 0].unsqueeze(0).repeat(B, 1, 1)

        # Audio-Visual Cloud Splatting (AVCS)
        # Anchor Projection
        camera_center = cam_pose[:, :3]
        camera_R = cam_pose[:, 3:].reshape(B, 3, 3)
        dir_pp = (self._xyz.repeat(B, 1, 1) - camera_center.unsqueeze(1).repeat(1, N_points, 1)).float()
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=-1, keepdim=True)
        scene_emb = self.pcd_av_proj(shs_view).reshape(B, N_points, -1) # B, N, C
        world2view_R = camera_R.unsqueeze(1).repeat(1, N_points, 1, 1).float()
        source_proj = (dir_pp.unsqueeze(2) @ world2view_R.transpose(2, 3)).squeeze(2) / self.max_norm
        ori_emb = self.vec_proj(self.vec_embedder(source_proj.reshape(-1, 3))).reshape(B, N_points, -1) # B, N, C

        if self.model_type == "simple-sh":
            # Simple version of Visual-to-Audio Splatting Transformer
            diff_attn_weights = torch.bmm(self.freq_weights.repeat(B, 1, 1), (scene_emb + ori_emb).permute(0, 2, 1)) / math.sqrt(self.d_model)
            mix_feat = (diff_attn_weights.flatten(0, 1) @ self._audio_features_f.flatten(-2, -1)).reshape(B, self.freq_num, -1)  / math.sqrt(N_points)
            hs = torch.bmm(diff_attn_weights, ori_emb).reshape(B, self.freq_num, -1)  / math.sqrt(N_points) # B F C
        else:
            # Visual-to-Audio Splatting Transformer
            freq = torch.linspace(-0.99, 0.99, self.freq_num, device=device).unsqueeze(1)
            freq_emb = self.freq_embedder(freq)    
            query_embed = self.query_prj(freq_emb).reshape(1, self.freq_num, 1, -1).repeat(1, 1, B, 1)
            hs, attn_weights = self.decoder(query_embed.flatten(0, 1), ori_emb.permute(1, 0, 2), memory_key_padding_mask=None,
                            pos=scene_emb.permute(1, 0, 2))
            mix_feat = (attn_weights.flatten(0, 1) @ self._audio_features_f.flatten(-2, -1)).reshape(B, self.freq_num, -1)
            hs = hs.reshape(1, self.freq_num, B, -1).permute(2, 0, 1, 3) # B, 1, freq_num, C

        mag_mask = self.mix_prj(mix_feat).expand(B, self.freq_num, self.time_num)
        diff = 2*self.diff_prj(hs).sigmoid() - 1
        diff = diff.reshape(B, self.freq_num, 1).expand(B, self.freq_num, self.time_num)

        # Spatial Audio Render Head (SARH)
        if self.model_type == "simple-sh" or self.model_type == "sh":
            x = ref_mag
            padding = self.dilation * (self.kernel_size - 1)
            attn_weights = eval_sh(1, self.time_freq_weights, dir_pp_normalized) # B, N, F
            time_attn_weights = torch.bmm(attn_weights, self.time_weights.repeat(B, 1, 1)) / math.sqrt(self.freq_num)
            weight = (time_attn_weights.permute(0, 2, 1).flatten(0, 1) @ self._acoustic_features_t).reshape(B, self.time_num, -1)[..., :x.shape[-1]] / math.sqrt(N_points)
            bias = (attn_weights.permute(0, 2, 1).flatten(0, 1) @ self._acoustic_features_t).reshape(B, self.freq_num, -1)[..., x.shape[-1]:] / math.sqrt(N_points)
            mix_mag = torch.bmm(mag_mask * x, weight.permute(0, 2, 1))+bias
        else:
            feat = hs.reshape(B, self.freq_num, -1).permute(0, 2, 1)
            mag_in = mag_mask * ref_mag
            mix_mag = self.post_process(mag_in.permute(0, 2, 1), feat).permute(0, 2, 1)

        left = (1 - diff) 
        right = (1 + diff)
        left_in = torch.stack([mix_mag, left * mag_mask * ori_ref_mag, mag_mask, -diff], 1)
        right_in = torch.stack([mix_mag, right * mag_mask * ori_ref_mag, mag_mask, diff], 1)
        left_mag_in = self.out(left_in).squeeze(1)
        right_mag_in = self.out(right_in).squeeze(1)
        reconstructed_stft_left = torch.polar(F.relu(left_mag_in+left * mag_mask * ori_ref_mag), left_phase)
        reconstructed_stft_right = torch.polar(F.relu(right_mag_in+right * mag_mask * ori_ref_mag), right_phase)
        reconstructed_stft = torch.stack([reconstructed_stft_left, reconstructed_stft_right]).permute(1, 0, 2, 3).flatten(0, 1)
        out_pred_wav = torch.istft(reconstructed_stft, n_fft=n_fft, hop_length=hop_length, win_length=window_length, window=torch_window, center=True).reshape(B, 2, -1)
        
        return out_pred_wav



def build_model(cfg, gaussian_model, scene):
    model = AVCloud(cfg, gaussian_model, scene)
    return model