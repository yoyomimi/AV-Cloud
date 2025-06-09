"""
Code of "AV-Cloud: Spatial Audio Rendering Through Audio-Visual Cloud Splatting" 

Copyright (c) 2023-2024 University of Washington. 

Developed in UW NeuroAI Lab by Mingfei Chen (lasiafly@uw.edu).
"""


import datetime
import logging
import math
import os
import sys
import time

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboardX import SummaryWriter

import libs.utils.misc as utils
from libs.evaluators.gen_eval import Evaluator
from libs.utils.utils import save_checkpoint


def data_loop(data_loader):
    """
    Loop an iterable infinitely
    """
    while True:
        for x in iter(data_loader):
            yield x


class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 criterion,
                 optimizer,
                 lr_scheduler,
                 logger,
                 log_dir,
                 performance_indicator='mse',
                 last_iter=-1,
                 rank=0,
                 device='cuda'):
        self.cfg = cfg
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        if log_dir:
            self.log_dir = os.path.join(log_dir, self.cfg.output_dir)
            self.epoch = last_iter + 1
        self.PI = performance_indicator
        self.rank = rank
        self.best_performance = 0.0
        self.is_best = False
        self.max_epoch = self.cfg.train.max_epoch
        self.model_name = self.cfg.model.file
        self.device = device
        self.iter_count = 0
        if self.optimizer is not None and rank == 0:
            self.writer = SummaryWriter(self.log_dir, comment=f'_rank{rank}')
            logging.info(f"max epochs = {self.max_epoch} ")
        self.evaluator = Evaluator(self.cfg, self.model_name, sampling_rate=self.cfg.dataset.sr)

    def _read_inputs(self, batch):
        for k in range(len(batch)):
            if isinstance(batch[k], tuple) or isinstance(batch[k], list):
                batch[k] = [b.to(self.device) for b in batch[k]]
            if isinstance(batch[k], dict):
                batch[k] = {key: value.to(self.device) for key, value in batch[k].items()}
            else:
                try:
                    batch[k] = batch[k].to(self.device)
                except:
                    pass
        return batch

    def _forward(self, data):
        index, cam_pose, gt_waveform, source_gt_audio = data
        pred_wav = self.model(cam_pose, source_gt_audio)
        if pred_wav is not None:
            min_len = min(pred_wav.shape[-1], gt_waveform.shape[-1])
            gt_waveform = gt_waveform[..., :min_len]
            pred_wav = pred_wav[..., :min_len]
            loss = self.criterion(pred_wav, gt_waveform, None)
        return loss

    def train(self, train_loader, eval_loader, save_dir=None):
        start_time = time.time()
        self.model.train()
        self.criterion.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(self.epoch)
        print_freq = self.cfg.train.print_freq
        eval_data_iter = data_loop(eval_loader)
        if self.epoch > self.max_epoch:
            logging.info("Optimization is done !")
            sys.exit(1)
        for data in metric_logger.log_every(train_loader, print_freq, header, self.logger):
            data = self._read_inputs(data)
            loss_dict = self._forward(data)
            losses = sum(loss_dict[k] for k in loss_dict.keys())
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_value = sum(loss_dict_reduced.values()).item()

            if not math.isfinite(loss_value) and self.rank == 0:
                print(loss_dict, loss_dict_reduced)
                self.logger.info("Loss is {}, stopping training".format(loss_value))
                import pdb; pdb.set_trace()
                self.optimizer.zero_grad()
                sys.exit(1)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            if not math.isfinite(sum(p.sum() for p in self.model.module.parameters() if p.requires_grad)):
                import pdb; pdb.set_trace()
            metric_logger.update(loss=loss_value, **loss_dict_reduced)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
            self.iter_count += 1
            # quick val
            if self.rank == 0 and self.iter_count % self.cfg.train.valiter_interval == 0:
                # evaluation
                if self.cfg.train.val_when_train:
                    self.quick_val(eval_data_iter)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': self.epoch, 'iter': self.iter_count}
        if self.rank == 0:
            for (key, val) in log_stats.items():
                self.writer.add_scalar(key, val, log_stats['iter'])
        self.lr_scheduler.step()

        # save checkpoint
        if self.rank == 0 and self.epoch >= 0 and self.epoch % self.cfg.train.save_interval == 0:
            # save checkpoint
            try:
                state_dict = self.model.module.state_dict()  # remove prefix of multi GPUs
            except AttributeError:
                state_dict = self.model.state_dict()

            if self.rank == 0:
                if self.cfg.train.save_every_checkpoint:
                    filename = f"{self.epoch}.pth"
                else:
                    filename = "latest.pth"
                save_dir = os.path.join(self.log_dir, self.cfg.output_dir)
                save_checkpoint(
                    {
                        'epoch': self.epoch,
                        'model': self.model_name,
                        'state_dict': state_dict,
                        'optimizer': self.optimizer.state_dict(),
                        'lr_scheduler': self.lr_scheduler.state_dict(),
                    },
                    self.is_best,
                    save_dir,
                    filename=f'{filename}'
                )
                pths = [
                    int(pth.split('.')[0]) for pth in os.listdir(save_dir)
                    if pth != 'latest.pth' and pth != 'model_best.pth'
                ]
                if len(pths) > 2:
                    os.system('rm {}'.format(
                        os.path.join(save_dir, '{}.pth'.format(min(pths)))))
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # print('Training time {}'.format(total_time_str))
        if self.rank == 0:
            self.logger.info('Training time {}'.format(total_time_str))
        self.epoch += 1
        # Ensure all processes synchronize at this point
        return True

    def quick_val(self, eval_data_iter):
        self.model.eval()
        self.criterion.eval()
        val_stats = {}
        plot_stats = {}
        with torch.no_grad():
            val_data = next(eval_data_iter)
            val_data = self._read_inputs(val_data)
            index, cam_pose, gt_waveform, source_gt_audio = val_data
            tar_out = gt_waveform.flatten(0, 1)
            pred_wav = self.model(cam_pose, source_gt_audio, is_val=True)
            B = len(tar_out)
            idx = np.random.choice(range(B))
            min_len = min(pred_wav.shape[-1], tar_out.shape[-1])
            tar_out = tar_out[..., :min_len]
            gt_waveform = gt_waveform[..., :min_len]
            pred_wav = pred_wav[..., :min_len]
            
            loss_dict = self.criterion(pred_wav, gt_waveform)

            pred_wav = pred_wav.reshape(-1, pred_wav.shape[-1])
            plot_stat = self.process_img(pred_wav[idx], tar_out[idx], sr=self.cfg.dataset.sr)
            plot_stats.update(plot_stat)
            gt_ir = tar_out[..., :pred_wav.shape[-1]]
            np_pred_ir = pred_wav.cpu().numpy()
            np_gt_ir = gt_ir.cpu().numpy()
            self.evaluator.evaluate(np_pred_ir, np_gt_ir, sr=44100)
            loss_stats = utils.reduce_dict(loss_dict)
            loss_stats = loss_dict
            for k, v in loss_stats.items():
                val_stats.setdefault(k, 0)
                val_stats[k] += v
            result = {}
            for key in self.evaluator.metrics.keys():
                result[key] = self.evaluator.metrics[key][-1]
            val_stats.update(result)

        # save metrics and loss
        log_stats = {**{f'eval_{k}': v for k, v in val_stats.items()},
                     'epoch': self.epoch, 'iter': self.iter_count}
        for (key, val) in log_stats.items():
            self.writer.add_scalar(key, val, log_stats['iter'])

        if plot_stats is not None:
            pattern = 'val_iter/{}'
            for k, v in plot_stats.items():
                self.writer.add_figure(pattern.format(k), v, log_stats['iter'])
                v.savefig(f'{self.cfg.output_dir}_{k}.jpg')

        if self.rank == 0:
            msg = ''
            for key, value in result.items():
                msg += f'{key}: {value:.4f}, '
            self.logger.info(msg)

        self.model.train()
        self.criterion.train()

    @staticmethod
    def process_img(pred_ir, gt_ir, sr=44100):
        gt_ir_plot = gt_ir.reshape(-1).data.cpu().numpy()
        pred_ir_plot = pred_ir.reshape(-1).data.cpu().numpy()
        min_val = np.minimum(np.min(gt_ir_plot), np.min(pred_ir_plot))
        max_val = np.maximum(np.max(gt_ir_plot), np.max(pred_ir_plot))
        gt_spec = librosa.stft(gt_ir_plot, n_fft=512, win_length=512, hop_length=128)
        pred_spec = librosa.stft(pred_ir_plot, n_fft=512, win_length=512, hop_length=128)
        fig = plt.figure()
        ax_gen = fig.add_subplot(3, 1, 1)
        ax_gen.plot(gt_ir_plot, color='green')
        ax_gen.plot(pred_ir_plot, color='red', alpha=0.7)
        ax_gen.set_ylim(min_val, max_val)
        ax_gt_ir_spec = fig.add_subplot(3, 1, 2)
        gt_spec_img = librosa.display.specshow(librosa.amplitude_to_db(abs(gt_spec)), sr=sr, hop_length=128, x_axis='time', y_axis='log', ax=ax_gt_ir_spec, cmap='coolwarm')
        ax_pred_ir_spec = fig.add_subplot(3, 1, 3)
        pred_spec_img = librosa.display.specshow(librosa.amplitude_to_db(abs(pred_spec)), sr=sr, hop_length=128, x_axis='time',
                                 y_axis='log', ax=ax_pred_ir_spec, cmap='coolwarm')
        fig.colorbar(gt_spec_img, ax=[ax_gt_ir_spec, ax_pred_ir_spec])
        fig.tight_layout()

        return {'fig_plot': fig}