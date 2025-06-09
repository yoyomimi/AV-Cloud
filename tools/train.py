"""
Code of "AV-Cloud: Spatial Audio Rendering Through Audio-Visual Cloud Splatting" 

Copyright (c) 2023-2024 University of Washington. 

Developed in UW NeuroAI Lab by Mingfei Chen (lasiafly@uw.edu).
"""


from __future__ import division, print_function, with_statement

import argparse
import os
import random
from importlib import import_module as impm

import _init_paths
import numpy as np
import torch
import torch.distributed as dist

from configs import cfg, update_config
from libs.utils import misc
from libs.utils.lr_scheduler import ExponentialLR
from libs.utils.utils import create_logger, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Neural Acoustic')
    parser.add_argument(
        '--cfg',
        dest='yaml_file',
        help='experiment configure file name, e.g. configs/base_config.yaml',
        required=True,
        type=str)
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
    parser.add_argument("--local_rank", type=int, default=0)
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


def get_ip(ip_addr):
    ip_list = ip_addr.split('-')[2:6]
    for i in range(4):
        if ip_list[i][0] == '[':
            ip_list[i] = ip_list[i][1:].split(',')[0]
    return f'tcp://{ip_list[0]}.{ip_list[1]}.{ip_list[2]}.{ip_list[3]}:2456'


def main_per_worker(rank, world_size, *args):
    args = parse_args()

    update_config(cfg, args)
    ngpus_per_node = torch.cuda.device_count()
    print('num gpu', ngpus_per_node)
    # torch seed
    seed = cfg.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '15377'


    proc_rank = rank
    local_rank = rank
    args.world_size = world_size

    args.distributed = (args.world_size > 1 or args.distributed)

    # create logger
    if proc_rank == 0:
        logger, output_dir = create_logger(cfg, proc_rank)
    else:
        logger = None

    train_dataset, eval_dataset = getattr(impm(cfg.dataset.name), 'build_dataset')(cfg)
    
    # distribution
    if args.distributed:
        dist.init_process_group(backend=cfg.dist_backend, init_method='env://', world_size=world_size, rank=rank)
        torch.distributed.barrier()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda")
        
        model = getattr(impm(cfg.model.file), 'build_model')(cfg)
        model.to(device)
        
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
        )
        batch_size = cfg.dataset.img_num_per_gpu
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset
        )
    else:
        assert proc_rank == 0, ('proc_rank != 0, it will influence '
                                'the evaluation procedure')
        if cfg.device == 'cuda':
            torch.cuda.set_device(local_rank)
        device = torch.device(cfg.device)
        model = getattr(impm(cfg.model.file), 'build_model')(cfg, train_dataset.gaussians, train_dataset.scene)
        model = torch.nn.DataParallel(model).to(device)
        train_sampler = None
        if ngpus_per_node == 0:
            batch_size = cfg.dataset.img_num_per_gpu
        else:
            batch_size = cfg.dataset.img_num_per_gpu * ngpus_per_node

    print('batch size:', batch_size)


    eval_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(True if train_sampler is None else False),
        drop_last=cfg.dataset.train.drop_last,
        num_workers=cfg.workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=True,
        drop_last=cfg.dataset.test.drop_last,
        num_workers=0,
        sampler=eval_sampler,
    )

    model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters()
                    if p.requires_grad]},
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    lr_scheduler = ExponentialLR(optimizer, decay_epochs=cfg.train.decay_epochs,
                                 gamma=cfg.train.gamma)
    
    model, optimizer, lr_scheduler, last_iter = load_checkpoint(cfg, model, optimizer, lr_scheduler, device)
    

    criterion = getattr(impm(cfg.train.criterion_file), 'Criterion')(cfg)

    Trainer = getattr(impm(cfg.train.file), 'Trainer')(
        cfg,
        model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        logger=logger,
        log_dir=cfg.log_dir,
        performance_indicator=cfg.pi,
        last_iter=last_iter,
        rank=proc_rank,
        device=device,
    )

    print('start training...')
    while True:
        Trainer.train(train_loader, eval_loader)


if __name__ == '__main__':
    main_per_worker(0, 1)