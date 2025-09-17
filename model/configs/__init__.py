from yacs.config import CfgNode as CN

# Define the default configuration using YACS CfgNode
cfg = CN()

cfg.epochs = 100
cfg.batch_size = 16
cfg.num_workers = 16
cfg.dataset_ratio = 1.
# seed for every thing
# include dataset split
cfg.seed = 42
cfg.scale = 1.
cfg.use_vqgan = False

cfg.gradient_clip_val = None
cfg.gradient_clip_algorithm = None


# VQ-GAN model configuration
cfg.vq_gan_model = CN()
cfg.vq_gan_model.embed_dim = 3
cfg.vq_gan_model.n_embed = 1024

# Decoder configuration
cfg.vq_gan_model.ddconfig = CN()
cfg.vq_gan_model.ddconfig.double_z = False
cfg.vq_gan_model.ddconfig.z_channels = 3
cfg.vq_gan_model.ddconfig.resolution = 128
cfg.vq_gan_model.ddconfig.in_channels = 3
cfg.vq_gan_model.ddconfig.out_ch = 3
cfg.vq_gan_model.ddconfig.ch = 128
cfg.vq_gan_model.ddconfig.ch_mult = [1, 1, 2, 4]  # representation shape 16x20
cfg.vq_gan_model.ddconfig.num_res_blocks = 2
cfg.vq_gan_model.ddconfig.attn_resolutions = [16]
cfg.vq_gan_model.ddconfig.dropout = 0.0

# Loss configuration
cfg.vq_gan_model.lossconfig = CN()
cfg.vq_gan_model.lossconfig.target = "UniT.taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator"
cfg.vq_gan_model.lossconfig.params = CN()
cfg.vq_gan_model.lossconfig.params.disc_conditional = False
cfg.vq_gan_model.lossconfig.params.disc_in_channels = 3
cfg.vq_gan_model.lossconfig.params.disc_start = 10000
cfg.vq_gan_model.lossconfig.params.disc_weight = 0.8
cfg.vq_gan_model.lossconfig.params.codebook_weight = 1.0


cfg.teacher_encoders = CN()
cfg.teacher_encoders.disp_path = ''
cfg.teacher_encoders.stress_path = ''
cfg.teacher_encoders.stress2_path = ''
cfg.teacher_encoders.cnorm_path = ''
cfg.teacher_encoders.area_shear_path = ''

cfg.scales = CN()
cfg.scales.disp = 1.
cfg.scales.depth = 1.
cfg.scales.stress = 1.
cfg.scales.stress2 = 1.
cfg.scales.cnorm = 1.
cfg.scales.area_shear = 1.

cfg.unit_scales = CN()
cfg.unit_scales.disp = 1.
cfg.unit_scales.depth = 1.
cfg.unit_scales.stress = 1.
cfg.unit_scales.stress2 = 1.
cfg.unit_scales.cnorm = 1.
cfg.unit_scales.area_shear = 1.

cfg.loss = CN()
cfg.loss.stress_weight = 1.
cfg.loss.disp_weight = 1.
cfg.loss.depth_weight = 1.
cfg.loss.cnorm_weight = 1.
cfg.loss.stress2_weight = 1.
cfg.loss.area_shear_weight = 1.


cfg.model = CN()
cfg.model.name = "DPT"
cfg.model.pretrained_model = ""
cfg.model.img_size = 256
cfg.model.patch_size = 16
cfg.model.in_chans = 6
cfg.model.encoder = 'vitl'
cfg.model.out_chans = [1]
cfg.model.loss = "L1"
cfg.model.backbone = ""
cfg.model.imagenet_pretrained = False
cfg.model.LoRA = False
cfg.model.LoRA_rank = 4
cfg.model.LoRA_scale = 1

cfg.model.cnn = CN()
cfg.model.cnn.decoder_mid_dim = [1024, 1024, 512, 256, 64]
cfg.model.cnn.decoder_output_dim = [1024, 512, 256, 128, 64]

# parameters for Hiera model
cfg.model.hiera = CN()
# this is the default for encoder
cfg.model.hiera.embed_dim = 96
cfg.model.hiera.num_heads = 1
cfg.model.hiera.stages = [2, 3, 16, 3]
cfg.model.hiera.q_pool = 2
cfg.model.hiera.patch_stride = [4, 4]
cfg.model.hiera.mlp_ratio = 4.0

cfg.model.hiera.decoder = "DPT"
# for DPT Decoder Head
cfg.model.hiera.use_bn = False
cfg.model.hiera.activation = "leaky_relu"
cfg.model.hiera.decoder_embed_dim = 256
cfg.model.hiera.decoder_mapping_channels = [256, 512, 1024]
cfg.model.hiera.decoder_depth_per_stage = 4

# for Vanilla model (decoder = "Vanilla")
cfg.model.hiera.decoder_num_heads = 1
cfg.model.hiera.decoder_depth = 3


# returning the encoder output in hiera
cfg.model.hiera.return_encoder_output = False

cfg.dataset = CN()
cfg.dataset.output_type = []
cfg.dataset.contiguous_on_direction = False
cfg.dataset.normalization = False

cfg.optimizer = CN()
cfg.optimizer.name = "Adam"
cfg.optimizer.lr = 1e-4
cfg.optimizer.eta_min = 1e-6

cfg.scheduler = CN()
# choose from cosine, linear_cosine
cfg.scheduler.name = "cosine"
# steps for linear warmup 
cfg.scheduler.warmup = 2000
# cycles for trainig (2k + 1)
cfg.scheduler.cycle_k = 0

cfg.metric = CN()
# value under this will be considered as negative
cfg.metric.PN_thresh = 0.0001
# rel error ratio (%) under this will be considered as true positive
cfg.metric.TF_rel_error_rate = 1.
# abs error ratio (m) under this will be considered as true positive
cfg.metric.TF_abs_error_thresh = 0.001
# For the curve
cfg.metric.num_bins = 10


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return cfg.clone()