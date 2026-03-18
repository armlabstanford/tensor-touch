import sys
import os
import numpy as np

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

from configs import get_cfg_defaults

LOCAL_CONFIG_PATH = 'model/configs/dt_ultra.yaml'
LOCAL_CONFIG_LARGE_PATH = 'model/configs/dt_ultra_large.yaml'

def denormalize_output(pred, cfg, output_names):
    """Denormalize model predictions following train_ViT_lightning.py"""
    pred_denorm = pred.clone()

    for idx, name in enumerate(output_names):
        scale = 1
        if 'disp' in name:
            scale = cfg.scales.disp
        elif 'stress1' in name:
            scale = cfg.scales.stress
        elif 'stress2' in name:
            scale = cfg.scales.stress2
        elif 'depth' in name:
            scale = cfg.scales.depth
        elif 'cnorm' in name:
            scale = cfg.scales.cnorm
        elif 'shear' in name:
            scale = cfg.scales.area_shear

        pred_denorm[:, idx, :, :] /= scale

    return pred_denorm

def get_output_names(cfg):
    """Get output names following train_ViT_lightning.py logic"""
    output_names = []
    dataset_output_type = cfg.dataset.output_type

    if cfg.dataset.contiguous_on_direction:
        if "depth" in dataset_output_type:
            output_names.append("depth")
        for d in ["x", "y", "z"]:
            for t in dataset_output_type:
                if t == "depth":
                    continue
                else:
                    output_names.append(f"{t}_{d}")
    else:
        for t in dataset_output_type:
            if t == "depth":
                output_names.append("depth")
            else:
                for d in ["x", "y", "z"]:
                    output_names.append(f"{t}_{d}")

    return output_names


def load_tactile_input():
    """Load deformed/undeformed tactile images and return 6-channel tensor."""
    deformed_pil = Image.open('model/deformed_tactile.png').convert('RGB')
    undeformed_pil = Image.open('model/undeformed_tactile.png').convert('RGB')

    to_tensor = transforms.ToTensor()
    deformed_tensor = to_tensor(deformed_pil)
    undeformed_tensor = to_tensor(undeformed_pil)

    X = torch.cat([deformed_tensor, undeformed_tensor], dim=0)  # (6, 256, 256)
    return X


def visualize_outputs(output_vis, output_names, title, save_path):
    """Visualize all output channels in a grid."""
    num_channels = output_vis.shape[0]
    cols = 5
    rows = (num_channels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else axes

    for i in range(num_channels):
        ax = axes[i]
        if 'depth' in output_names[i]:
            im = ax.imshow(output_vis[i], cmap='viridis', vmin=0, vmax=output_vis[i].max())
        elif 'stress' in output_names[i] or 'disp' in output_names[i]:
            vmax = max(abs(output_vis[i].min()), abs(output_vis[i].max()))
            im = ax.imshow(output_vis[i], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(output_vis[i], cmap='viridis')

        ax.set_title(f'{output_names[i]}\n[{output_vis[i].min():.3f}, {output_vis[i].max():.3f}]')
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.8)

    for i in range(num_channels, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


def run_model(model, X_input, cfg, model_name, save_path):
    """Run inference and visualize outputs for a given model."""
    output_names = get_output_names(cfg)
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Output names: {output_names}")

    model.eval()
    X = X_input.unsqueeze(0).cuda()
    print(f"Input shape: {X.shape}")

    # Encoder embedding
    z, _ = model.encoder(X, None, return_intermediates=True)
    print(f"Encoder output (vision tokens): {z.shape}")

    with torch.no_grad():
        output = model(X)
        print(f"Raw output shape: {output.shape}")

        output_denorm = denormalize_output(output, cfg, output_names)

    output_vis = output_denorm[0].detach().cpu().numpy()
    visualize_outputs(output_vis, output_names, model_name, save_path)


if __name__ == '__main__':
    X_input = load_tactile_input()
    print(f"Tactile input shape: {X_input.shape}")
    print(f"Value range: [{X_input.min():.3f}, {X_input.max():.3f}]")

    # --- Hiera Base (original, from torch hub) ---
    print("\nLoading Hiera Base...")
    base_model = torch.hub.load(
        'peasant98/DenseTact-Model', 'hiera',
        pretrained=True, map_location='cpu', trust_repo=True
    ).cuda()

    cfg_base = get_cfg_defaults()
    cfg_base.merge_from_file(LOCAL_CONFIG_PATH)

    run_model(base_model, X_input, cfg_base,
              "Hiera Base (original)",
              "tactile_output_base.png")

    del base_model
    torch.cuda.empty_cache()

    # --- Hiera Large v1 (epoch 22, PSNR 45.32) ---
    print("\nLoading Hiera Large v1...")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from hubconf import hiera_large_v1, hiera_large_v2

    large_v1 = hiera_large_v1(pretrained=True, map_location='cpu').cuda()

    cfg_large = get_cfg_defaults()
    cfg_large.merge_from_file(LOCAL_CONFIG_LARGE_PATH)

    run_model(large_v1, X_input, cfg_large,
              "Hiera Large v1 (epoch 22, PSNR 45.32)",
              "tactile_output_large_v1.png")

    del large_v1
    torch.cuda.empty_cache()

    # --- Hiera Large v2 (epoch 43, PSNR 52.22) ---
    print("\nLoading Hiera Large v2...")
    large_v2 = hiera_large_v2(pretrained=True, map_location='cpu').cuda()

    run_model(large_v2, X_input, cfg_large,
              "Hiera Large v2 (epoch 43, PSNR 52.22)",
              "tactile_output_large_v2.png")

    print(f"\n{'='*60}")
    print("All models tested successfully!")
    print("Visualizations saved:")
    print("  - tactile_output_base.png")
    print("  - tactile_output_large_v1.png")
    print("  - tactile_output_large_v2.png")
