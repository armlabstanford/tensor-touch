# hubconf.py
# PyTorch Hub config
dependencies = ["torch", "yacs", "torchvision", "timm"]
__all__ = ["hiera"]

import os
import torch
from torch.hub import load_state_dict_from_url
from configs import get_cfg_defaults
from models import build_model  # , replace_LoRA  # if you need the function

_WEIGHTS = {
    "base": "https://github.com/peasant98/DenseTact-Model/releases/download/models/best.ckpt"
}

# Get absolute path to config relative to hubconf.py location
HUBCONF_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_CONFIG_PATH = os.path.join(HUBCONF_DIR, "configs", "dt_ultra.yaml")

def hiera(pretrained: bool = True, variant: str = "base", map_location="cpu", **kwargs):
    """
    Build the model and (optionally) load pretrained weights.

    Args:
        pretrained: load weights from _WEIGHTS table
        variant: key into _WEIGHTS
        map_location: torch.load map_location
    """
    cfg = get_cfg_defaults()
    cfg.merge_from_file(LOCAL_CONFIG_PATH)
    model = build_model(cfg)

    # If your model defines this method; otherwise call a helper function instead.
    if hasattr(model, "replace_LoRA"):
        model.replace_LoRA(4, 1)

    if pretrained:
        state = load_state_dict_from_url(
            _WEIGHTS[variant],
            map_location=map_location,
            check_hash=False,  # set True only if filename includes a hash
        )

        # Handle Lightning .ckpt
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # Strip common wrappers
        cleaned = {k.replace("model.", "").replace("module.", ""): v for k, v in state.items()}

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing or unexpected:
            print(f"[hub] missing keys: {missing}\n[hub] unexpected keys: {unexpected}")

    return model
