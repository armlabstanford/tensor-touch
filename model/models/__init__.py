from .dt_vit import dt_vit_base_patch16, dt_vit_large_patch16, dt_vit_huge_patch14
from .dpt import DPTV2Net, HieraDPT
from .dense import DTDenseNet, DTNet
from .siamese_vit import SiameseViT, siamese_vit_base_patch16, siamese_vit_large_patch16

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
    elif cfg.model.name == "SiameseViT":
        model = SiameseViT(
            img_size=cfg.model.siamese_vit.img_size,
            patch_size=cfg.model.siamese_vit.patch_size,
            in_chans=cfg.model.siamese_vit.in_chans,
            out_chans=cfg.model.out_chans,
            embed_dim=cfg.model.siamese_vit.embed_dim,
            depth=cfg.model.siamese_vit.depth,
            num_heads=cfg.model.siamese_vit.num_heads,
            decoder_embed_dim=cfg.model.siamese_vit.decoder_embed_dim,
            decoder_depth=cfg.model.siamese_vit.decoder_depth,
            decoder_num_heads=cfg.model.siamese_vit.decoder_num_heads,
            mlp_ratio=cfg.model.siamese_vit.mlp_ratio,
            fusion_method=cfg.model.siamese_vit.fusion_method,
        )
    elif cfg.model.name == "DinoV2":
        pass
    else:
        raise NotImplementedError("Model not implemented {}".format(cfg.model.name))

    return model

