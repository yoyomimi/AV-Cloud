from yacs.config import CfgNode as CN

cfg = CN()

cfg.device = 'cuda'

cfg.dist_backend = 'nccl'

cfg.log_dir = 'logs/'
cfg.output_dir = 'outputs/'
cfg.result_dir = 'results/'

cfg.seed = 42

cfg.workers = 4

cfg.pi = 'psnr'

cfg.model = ''


# dataset
cfg.dataset = CN()

cfg.dataset.img_num_per_gpu = 1

cfg.dataset.N_points = 256
cfg.dataset.H = 1024
cfg.dataset.W = 1024
cfg.dataset.name = ''
cfg.dataset.video = ''
cfg.dataset.data_root = 'data/'
cfg.dataset.visible = False
cfg.dataset.sr = 44100

cfg.dataset.train = CN()

cfg.dataset.train.sampler = ''
cfg.dataset.train.drop_last = True
cfg.dataset.train.shuffle = True


cfg.dataset.test = CN()
cfg.dataset.test.sampler = ''
cfg.dataset.test.batch_sampler = ''
cfg.dataset.test.drop_last = False
cfg.dataset.test.shuffle = False

# preprocessing
cfg.preprocessing = CN()
cfg.preprocessing.audio = CN()
cfg.preprocessing.audio.sampling_rate = 16000
cfg.preprocessing.audio.max_wav_value = 32768
cfg.preprocessing.stft = CN()
cfg.preprocessing.stft.filter_length = 1024
cfg.preprocessing.stft.hop_length = 160
cfg.preprocessing.stft.win_length = 1024
cfg.preprocessing.mel = CN()
cfg.preprocessing.mel.n_mel_channels = 64
cfg.preprocessing.mel.mel_fmin = 0
cfg.preprocessing.mel.mel_fmax = 8000
cfg.preprocessing.mel.freqm = 0
cfg.preprocessing.mel.timem = 0
cfg.preprocessing.mel.blur = False
cfg.preprocessing.mel.mean = -4.63
cfg.preprocessing.mel.std = 2.74
cfg.preprocessing.mel.target_length = 1024



# model
cfg.model = CN()
cfg.model.file = ''
cfg.model.resume_path = ''
cfg.model.joint_emb_dim = 512
cfg.model.pretrained_encoder = ''
cfg.model.model_type = 'full'
cfg.model.render_type = 'base'


# train
cfg.train = CN()

cfg.train.file = 'BaseTrainer'

cfg.train.resume = False
cfg.train.criterion_file = 'BaseCriterion'
cfg.train.body_sample_ratio = 0.5
cfg.train.n_rays = 1024
cfg.train.n_samples = 64
cfg.train.ddim_steps = 200
cfg.train.ep_iter = 500
cfg.train.lr = 1e-4
cfg.train.lr_backbone = 1e-5
cfg.train.gamma = 0.1
cfg.train.decay_epochs = 1000
cfg.train.weight_decay = 0.0001
cfg.train.max_epoch = 1000

cfg.train.print_freq = 10
cfg.train.save_every_checkpoint = True
cfg.train.save_interval = 1
cfg.train.valiter_interval = 100
cfg.train.val_when_train = False

cfg.train.duration = 5


# test
cfg.test = CN()

cfg.test.save_imgs = True
cfg.test.is_vis = False


def update_config(config, args):
    config.defrost()
    # set cfg using yaml config file
    config.merge_from_file(args.yaml_file)
    # update cfg using args
    config.merge_from_list(args.opts)
    config.freeze()