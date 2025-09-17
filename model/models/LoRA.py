# LoRA Module
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from torch.optim import Adam
from tqdm import tqdm
import os

def calculateIoU(pred, gt):
    intersect = (pred * gt).sum(dim=(-1, -2))
    union = pred.sum(dim=(-1, -2)) + gt.sum(dim=(-1, -2)) - intersect
    ious = intersect.div(union)
    return ious

class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}"
            )

        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        return up_hidden_states.to(orig_dtype)

    @property
    def weight(self):
        return self.up.weight @ self.down.weight

    @property
    def bias(self):
        return 0

class MonkeyPatchLoRALinear(nn.Module):
    # It's "monkey patch" means you can replace nn.Linear with the new
    # LoRA Linear class without modifying any other code.
    def __init__(self, fc: nn.Linear, rank=4, lora_scale=1):
        super().__init__()
        if rank > min(fc.in_features, fc.out_features):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(fc.in_features, fc.out_features)}"
            )
        if not isinstance(fc, nn.Linear):
            raise ValueError(
                f"MonkeyPatchLoRALinear only support nn.Linear, but got {type(fc)}"
            )

        self.fc = fc
        self.rank = rank
        self.lora_scale = lora_scale

        in_features = fc.in_features
        out_features = fc.out_features
        self.fc_lora = LoRALinearLayer(in_features, out_features, rank)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc(hidden_states) + \
                        self.lora_scale * self.fc_lora(hidden_states)
        return hidden_states

    @property
    def weight(self):
        return self.fc.weight + self.lora_scale * self.fc_lora.weight

    @property
    def bias(self):
        return self.fc.bias

# your implementation

class LoRAConv2DLayer(nn.Module):
    def __init__(self, in_features, out_features, kernel, stride = 1, padding = 0, rank=4):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}"
            )
        
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel
        self.rank = rank

        self.down = nn.Conv2d(in_features, rank, 1, 1, 0, bias=False)
        self.up = nn.Conv2d(rank, out_features, kernel, stride, padding, bias=False)

        nn.init.normal_(self.down.weight, std = 1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        return up_hidden_states.to(orig_dtype)
    
    @property
    def weight(self):
        composite_weight = torch.einsum('rnjk,mrjk->mnjk', self.up.weight, self.down.weight)
        return composite_weight

    @property
    def bias(self):
        return 0

class MonkeyPatchLoRAConv2D(nn.Module):
    def __init__(self, module: nn.Conv2d, rank=4, lora_scale=1):
        super().__init__()
        if rank > min(module.in_channels, module.out_channels):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(module.in_channels, module.out_channels)}"
            )
        if not isinstance(module, nn.Conv2d):
            raise ValueError(
                f"MonkeyPatchLoRALinear only support nn.Linear, but got {type(module)}"
            )

        self.conv = module
        self.rank = rank
        self.kernel_size = module.kernel_size
        self.stride = module.stride
        self.padding = module.padding
        self.lora_scale = lora_scale

        in_channels = module.in_channels
        out_channels = module.out_channels
        self.conv_lora = LoRAConv2DLayer(in_channels, out_channels, self.kernel_size, self.stride, self.padding, rank)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(hidden_states) + \
                        self.lora_scale * self.conv_lora(hidden_states)
        return hidden_states
    
    @property
    def weight(self):
        return self.conv.weight + self.lora_scale * self.conv_lora.weight

    @property
    def bias(self):
        return self.conv.bias

# class LoRAConvTranspose2DLayer(nn.Module):
#     ...

# class MonkeyPatchLoRAConvTranspose2D(nn.Module):
#     ...

def replace_LoRA(model:nn.Module, cls, rank=4, lora_scale=1):
    for name, block in model.named_children():
        # patch every nn.Linear in Mlp
        if isinstance(block, nn.Linear) and cls == MonkeyPatchLoRALinear:
            block = cls(block, rank, lora_scale)
            setattr(model, name, block)

            for param in block.fc.parameters():
                param.requires_grad_(False)
            for param in block.fc_lora.parameters():
                param.requires_grad_(True)
        
        elif isinstance(block, nn.Conv2d) and cls == MonkeyPatchLoRAConv2D:
            min_channel = min(block.in_channels, block.out_channels)
            if min_channel > 4:
                block = cls(block, rank, lora_scale)
                setattr(model, name, block)

                for param in block.conv.parameters():
                    param.requires_grad_(False)
                for param in block.conv_lora.parameters():
                    param.requires_grad_(True)
                    
        else:
            replace_LoRA(block, cls)