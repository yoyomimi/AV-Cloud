
# We modified the Criterion code from https://github.com/facebookresearch/detr/blob/main/models/position_encoding.py

# This source code is licensed under Copyright 2020 - present, Facebook, Inc (https://github.com/facebookresearch/detr/blob/main/LICENSE)


import math

import numpy as np
import torch
import torch.nn as nn


class embedding_module_log(nn.Module):
    def __init__(self, funcs=[torch.sin, torch.cos], num_freqs=20, max_freq=10, ch_dim=1, include_in=True):
        super().__init__()
        self.functions = funcs
        self.num_functions = list(range(len(funcs)))
        self.freqs = torch.nn.Parameter(2.0**torch.from_numpy(np.linspace(start=0.0,stop=max_freq, num=num_freqs).astype(np.single)), requires_grad=False)
        self.ch_dim = ch_dim
        self.funcs = funcs
        self.include_in = include_in

    def forward(self, x_input):
        if self.include_in:
            out_list = [x_input]
        else:
            out_list = []
        for func in self.funcs:
            for freq in self.freqs:
                out_list.append(func(x_input*freq))
        return torch.cat(out_list, dim=self.ch_dim)



class PositionalEncoder(nn.Module):
    def __init__(self, input_dims=2, max_freq=8., N=64, log_sampling=False,):
        super().__init__()
        if log_sampling:
            bands = 2**torch.linspace(1, max_freq, steps=N, requires_grad=False, dtype=torch.float)
        else:
            bands = torch.linspace(1, 2**max_freq, steps=N, requires_grad=False, dtype=torch.float)
        self.bands = nn.Parameter(bands, requires_grad=False)
        self.input_dims = input_dims

    def output_dims(self):
        return self.input_dims * 2 * len(self.bands)

    def forward(self, x):
        assert(x.shape[-1] == self.input_dims)
        raw_freqs = torch.tensordot(x, self.bands, dims=0)
        raw_freqs = raw_freqs.reshape(x.shape[:-1] + (-1,))
        return torch.cat([raw_freqs.sin(), raw_freqs.cos()], dim=-1)



class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        # assert mask is not None
        # not_mask = ~mask
        # y_embed = not_mask.cumsum(1, dtype=torch.float32)
        # x_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = x[..., 0]
        y_embed = x[..., 1]
        # if self.normalize:
        #     eps = 1e-6
        #     y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        #     x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=-1).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=-1).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(cfg):
    N_steps = cfg.TRANSFORMER.HIDDEN_DIM // 2
    if cfg.TRANSFORMER.POSITION_EMBEDDING in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif cfg.TRANSFORMER.POSITION_EMBEDDING in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {cfg.TRANSFORMER.POSITION_EMBEDDING}")

    return position_embedding