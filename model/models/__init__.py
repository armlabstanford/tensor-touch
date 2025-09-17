from .dt_vit import dt_vit_base_patch16, dt_vit_large_patch16, dt_vit_huge_patch14
from .dpt import DPTV2Net, HieraDPT
from .dense import DTDenseNet, DTNet

from .model_mae import mae_vit_base_patch16, mae_vit_large_patch16
from .hiera_mae import mae_hiera_base_256, mae_hiera_base_plus_256, mae_hiera_large_256
from .LoRA import replace_LoRA, MonkeyPatchLoRALinear

pretrain_dict = {
    "mae_hiera_base_256": mae_hiera_base_256,
    "mae_hiera_base_plus_256": mae_hiera_base_plus_256,
    "mae_hiera_large_256": mae_hiera_large_256,
    "mae_vit_base_patch16": mae_vit_base_patch16,
    "mae_vit_large_patch16": mae_vit_large_patch16
}

def build_model(cfg):
    # build model
    if cfg.model.name == "ViT":
        model = dt_vit_base_patch16(img_size=cfg.model.img_size, 
                        in_chans=cfg.model.in_chans, out_chans=cfg.model.out_chans)
    elif cfg.model.name == "DPT":
        vit_model = dt_vit_base_patch16(img_size=cfg.model.img_size, 
                        in_chans=cfg.model.in_chans, out_chans=cfg.model.out_chans)
        model = DPTV2Net(img_size=cfg.model.img_size, encoder=vit_model,
                        in_chans=cfg.model.in_chans, out_dims=cfg.model.out_chans)
    elif cfg.model.name == "DenseNet":
        model = DTDenseNet(out_chans=[cfg.model.out_chans])
    elif cfg.model.name == "DenseNetV2":
        model = DTNet(cfg)
    elif cfg.model.name == "HieraDPT":
        model = HieraDPT(cfg)
    elif cfg.model.name == "DinoV2":
        pass
    else:
        raise NotImplementedError("Model not implemented {}".format(cfg.model.name))

    return model

