# hubconf.py
# PyTorch Hub config
dependencies = ["torch", "yacs", "torchvision", "timm"]
__all__ = ["hiera", "hiera_large_v1", "hiera_large_v2"]

import os
import torch
from torch.hub import load_state_dict_from_url
from configs import get_cfg_defaults
from models import build_model  # , replace_LoRA  # if you need the function

_WEIGHTS = {
    "base": "https://github.com/peasant98/DenseTact-Model/releases/download/models/best.ckpt",
    "large_v1": "https://github.com/peasant98/DenseTact-Model/releases/download/models/dt_model-epoch.22-val_psnr.45.32.ckpt",
    "large_v2": "https://github.com/peasant98/DenseTact-Model/releases/download/models/dt_model-epoch.43-val_psnr.52.22.ckpt",
}

# Get absolute path to config relative to hubconf.py location
HUBCONF_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_CONFIG_PATH = os.path.join(HUBCONF_DIR, "configs", "dt_ultra.yaml")
LOCAL_CONFIG_LARGE_PATH = os.path.join(HUBCONF_DIR, "configs", "dt_ultra_large.yaml")


def _load_and_clean_state_dict(state, map_location="cpu"):
    """Handle Lightning .ckpt format and strip common key prefixes."""
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {k.replace("model.", "").replace("module.", ""): v for k, v in state.items()}
    return cleaned


def _build_and_load(config_path, weights_key, pretrained, map_location):
    """Build model from config and optionally load pretrained weights."""
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_path)
    model = build_model(cfg)

    if hasattr(model, "replace_LoRA"):
        model.replace_LoRA(4, 1)

    if pretrained:
        state = load_state_dict_from_url(
            _WEIGHTS[weights_key],
            map_location=map_location,
            check_hash=False,
        )
        cleaned = _load_and_clean_state_dict(state)

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing or unexpected:
            print(f"[hub] missing keys: {missing}\n[hub] unexpected keys: {unexpected}")

    return model


def hiera(pretrained: bool = True, map_location="cpu", **kwargs):
    """
    Build the Hiera Base model and (optionally) load pretrained weights.

    Args:
        pretrained: load weights from GitHub release
        map_location: torch.load map_location
    """
    return _build_and_load(LOCAL_CONFIG_PATH, "base", pretrained, map_location)


def hiera_large_v1(pretrained: bool = True, map_location="cpu", **kwargs):
    """
    Hiera Large v1 — early training checkpoint (epoch 22, PSNR 45.32).

    Architecture: Hiera Large encoder (embed_dim=144, stages=[2,6,36,4])
    with QUpsampling decoder and LoRA fine-tuning.

    Args:
        pretrained: load weights from GitHub release
        map_location: torch.load map_location
    """
    return _build_and_load(LOCAL_CONFIG_LARGE_PATH, "large_v1", pretrained, map_location)


def hiera_large_v2(pretrained: bool = True, map_location="cpu", **kwargs):
    """
    Hiera Large v2 — best checkpoint (epoch 43, PSNR 52.22).

    Architecture: Hiera Large encoder (embed_dim=144, stages=[2,6,36,4])
    with QUpsampling decoder and LoRA fine-tuning.

    Args:
        pretrained: load weights from GitHub release
        map_location: torch.load map_location
    """
    return _build_and_load(LOCAL_CONFIG_LARGE_PATH, "large_v2", pretrained, map_location)
