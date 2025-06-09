"""
Code of "AV-Cloud: Spatial Audio Rendering Through Audio-Visual Cloud Splatting" 

Copyright (c) 2023-2024 University of Washington. 

Developed in UW NeuroAI Lab by Mingfei Chen (lasiafly@uw.edu).
"""


import logging
import math
import os
import os.path as osp
import time

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve


def sequence_generation(volume, duration, c, fs, max_rate=10000):

    # repeated constant
    fpcv = 4 * np.pi * c ** 3 / volume

    # initial time
    t0 = ((2 * np.log(2)) / fpcv) ** (1.0 / 3.0)
    times = [t0]
    while times[-1] < t0 + duration:

        # uniform random variable
        z = np.random.rand()
        # rate of the point process at this time
        mu = np.minimum(fpcv * (t0 + times[-1]) ** 2, max_rate)
        # time interval to next point
        dt = np.log(1 / z) / mu

        times.append(times[-1] + dt)

    # convert from continuous to discrete time
    indices = (np.array(times) * fs).astype(np.int)
    seq = np.zeros(indices[-1] + 1)
    seq[indices] = np.random.choice([1, -1], size=len(indices))

    return seq

def octave_bands(fc=1000, third=False, start=0.0, n=8):
    """
    Create a bank of octave bands
    Parameters
    ----------
    fc : float, optional
        The center frequency
    third : bool, optional
        Use third octave bands (default False)
    start : float, optional
        Starting frequency for octave bands in Hz (default 0.)
    n : int, optional
        Number of frequency bands (default 8)
    """

    div = 1
    if third:
        div = 3

    # Octave Bands
    fcentre = fc * (
        2.0 ** (np.arange(start * div, (start + n) * div - (div - 1)) / div)
    )
    fd = 2 ** (0.5 / div)
    bands = np.array([[f / fd, f * fd] for f in fcentre])

    return bands, fcentre


class OctaveBandsFactory(object):
    """
    A class to process uniformly all properties that are defined on octave
    bands.
    Each property is stored for an octave band.
    Attributes
    ----------
    base_freq: float
        The center frequency of the first octave band
    fs: float
        The target sampling frequency
    n_bands: int
        The number of octave bands needed to cover from base_freq to fs / 2
        (i.e. floor(log2(fs / base_freq)))
    bands: list of tuple
        The list of bin boundaries for the octave bands
    centers
        The list of band centers
    all_materials: list of Material
        The list of all Material objects created by the factory
    Parameters
    ----------
    base_frequency: float, optional
        The center frequency of the first octave band (default: 125 Hz)
    fs: float, optional
        The sampling frequency used (default: 16000 Hz)
    third_octave: bool, optional
        Use third octave bands if True (default: False)
    """

    def __init__(self, base_frequency=125.0, fs=16000, n_fft=512):

        self.base_freq = base_frequency
        self.fs = fs
        self.n_fft = n_fft

        # compute the number of bands
        self.n_bands = math.floor(np.log2(fs / base_frequency))

        self.bands, self.centers = octave_bands(
            fc=self.base_freq, n=self.n_bands, third=False
        )

        self._make_filters()

    def get_bw(self):
        """Returns the bandwidth of the bands"""
        return np.array([b2 - b1 for b1, b2 in self.bands])

    def analysis(self, x, band=None):
        """
        Process a signal x through the filter bank
        Parameters
        ----------
        x: ndarray (n_samples)
            The input signal
        Returns
        -------
        ndarray (n_samples, n_bands)
            The input signal filters through all the bands
        """

        if band is None:
            bands = range(self.filters.shape[1])
        else:
            bands = [band]

        output = np.zeros((x.shape[0], len(bands)), dtype=x.dtype)

        for i, b in enumerate(bands):
            output[:, i] = fftconvolve(x, self.filters[:, b], mode="same")

        if output.shape[1] == 1:
            return output[:, 0]
        else:
            return output

    def __call__(self, coeffs=0.0, center_freqs=None, interp_kind="linear", **kwargs):
        """
        Takes as input a list of values with optional corresponding center frequency.
        Returns a list with the correct number of octave bands. Interpolation and
        extrapolation are used to fill in the missing values.
        Parameters
        ----------
        coeffs: list
            A list of values to use for the octave bands
        center_freqs: list, optional
            The optional list of center frequencies
        interp_kind: str
            Specifies the kind of interpolation as a string (‘linear’,
            ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’,
            ‘next’, where ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a
            spline interpolation of zeroth, first, second or third order;
            ‘previous’ and ‘next’ simply return the previous or next value of
            the point) or as an integer specifying the order of the spline
            interpolator to use. Default is ‘linear’.
        """

        if not isinstance(coeffs, (list, np.ndarray)):
            # when the parameter is a scalar just do flat extrapolation
            ret = [coeffs] * self.n_bands

        if len(coeffs) == 1:
            ret = coeffs * int(self.n_bands)

        else:
            # by default infer the center freq to be the low ones
            if center_freqs is None:
                center_freqs = self.centers[: len(coeffs)]

            # create the interpolator in log domain
            interpolator = interp1d(
                np.log2(center_freqs),
                coeffs,
                fill_value="extrapolate",
                kind=interp_kind,
            )
            ret = interpolator(np.log2(self.centers))

            # now clip between 0. and 1.
            ret[ret < 0.0] = 0.0
            ret[ret > 1.0] = 1.0

        return ret

    def _make_filters(self):
        """
        Create the band-pass filters for the octave bands
        Parameters
        ----------
        order: int, optional
            The order of the IIR filters (default: 8)
        output: {'ba', 'zpk', 'sos'}
            Type of output: numerator/denominator ('ba'), pole-zero ('zpk'), or
            second-order sections ('sos'). Default is 'ba'.
        Returns
        -------
        A list of callables that will each apply one of the band-pass filters
        """

        """
        filter_bank = bandpass_filterbank(
            self.bands, fs=self.fs, order=order, output=output
        )
        return [lambda sig: sosfiltfilt(bpf, sig) for bpf in filter_bank]
        """

        # This seems to work only for Octave bands out of the box
        centers = self.centers
        n = len(self.centers)

        new_bands = [[centers[0] / 2, centers[1]]]
        for i in range(1, n - 1):
            new_bands.append([centers[i - 1], centers[i + 1]])
        new_bands.append([centers[-2], self.fs / 2])

        n_freq = self.n_fft // 2 + 1
        freq_resp = np.zeros((n_freq, n))
        freq = np.arange(n_freq) / self.n_fft * self.fs

        for b, (band, center) in enumerate(zip(new_bands, centers)):
            lo = np.logical_and(band[0] <= freq, freq < center)
            freq_resp[lo, b] = 0.5 * (1 + np.cos(2 * np.pi * freq[lo] / center))

            if b != n - 1:
                hi = np.logical_and(center <= freq, freq < band[1])
                freq_resp[hi, b] = 0.5 * (1 - np.cos(2 * np.pi * freq[hi] / band[1]))
            else:
                hi = center <= freq
                freq_resp[hi, b] = 1.0

        filters = np.fft.fftshift(
            np.fft.irfft(freq_resp, n=self.n_fft, axis=0),
            axes=[0],
        )

        # remove the first sample to make them odd-length symmetric filters

        self.filters = filters[1:, :]

        self.filters = filters[1:, :]


def resource_path(relative_path):
    """To get the absolute path"""
    base_path = osp.abspath(".")

    return osp.join(base_path, relative_path)


def ensure_dir(root_dir, rank=0):
    if not osp.exists(root_dir) and rank == 0:
        print(f'=> creating {root_dir}')
        os.mkdir(root_dir)
    else:
        while not osp.exists(root_dir):
            print(f'=> wait for {root_dir} created')
            time.sleep(10)

    return root_dir

def create_logger(cfg, rank=0):
    # working_dir root
    abs_working_dir = resource_path('work_dirs')
    working_dir = ensure_dir(abs_working_dir, rank)
    # output_dir root
    output_root_dir = ensure_dir(os.path.join(working_dir, cfg.output_dir), rank)
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    final_output_dir = ensure_dir(os.path.join(output_root_dir, time_str), rank)
    # set up logger
    logger = setup_logger(final_output_dir, time_str, rank)

    return logger, final_output_dir


def setup_logger(final_output_dir, time_str, rank, phase='train'):
    log_file = f'{phase}_{time_str}_rank{rank}.log'
    final_log_file = os.path.join(final_output_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def load_checkpoint(cfg, model, optimizer, lr_scheduler, device, module_name='model'):
    last_iter = -1
    resume_path = cfg.model.resume_path
    resume = cfg.train.resume
    if resume_path and resume:
        if osp.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location='cpu')
            # resume
            if 'state_dict' in checkpoint:
                model.module.load_state_dict(checkpoint['state_dict'], strict=False)
                logging.info(f'==> model pretrained from {resume_path} \n')
            elif 'model' in checkpoint:
                if module_name == 'detr':
                    model.module.detr_head.load_state_dict(checkpoint['model'], strict=False)
                    logging.info(f'==> detr pretrained from {resume_path} \n')
                else:
                    model.module.load_state_dict(checkpoint['model'], strict=False)
                    logging.info(f'==> model pretrained from {resume_path} \n')
            if 'optimizer' in checkpoint and optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
                logging.info(f'==> optimizer resumed, continue training')
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
                
            if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint and lr_scheduler is not None:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                last_iter = checkpoint['epoch']
                logging.info(f'==> last_epoch = {last_iter}')
            else:
                lr_dict = lr_scheduler.state_dict()
                lr_dict['last_epoch'] = checkpoint['epoch']
                lr_dict['_last_lr'] = [optimizer.param_groups[0]['lr']]
                lr_scheduler.load_state_dict(lr_dict)
            if 'epoch' in checkpoint and lr_scheduler is not None:
                last_iter = checkpoint['epoch']
                logging.info(f'==> last_epoch = {last_iter}')
            # pre-train
        else:
            logging.error(f"==> checkpoint do not exists: \"{resume_path}\"")
            raise FileNotFoundError
    else:
        logging.info("==> train model without resume")

    return model, optimizer, lr_scheduler, last_iter


def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth'):
    if not osp.exists(output_dir):
        os.mkdir(output_dir)
    torch.save(states, os.path.join(output_dir, filename))
    logging.info(f'save model to {output_dir}')
    if is_best:
        torch.save(states['state_dict'], os.path.join(output_dir, 'model_best.pth'))


def load_eval_model(resume_path, model):
    if resume_path != '':
        if osp.exists(resume_path):
            print(f'==> model load from {resume_path}')
            checkpoint = torch.load(resume_path)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            print(f"==> checkpoint do not exists: \"{resume_path}\"")
            raise FileNotFoundError
    return model


def write_dict_to_json(mydict, f_path):
    import json

    import numpy
    class DateEnconding(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
                numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
                numpy.uint16,numpy.uint32, numpy.uint64)):
                return int(obj)
            elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32, 
                numpy.float64)):
                return float(obj)
            elif isinstance(obj, (numpy.ndarray,)): # add this line
                return obj.tolist() # add this line
            return json.JSONEncoder.default(self, obj)
    with open(f_path, 'w') as f:
        json.dump(mydict, f, cls=DateEnconding)
        print("write down det dict to %s!" %(f_path))