import numpy as np
import cv2
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import sys
import os

# Add parent directory to path to import configs
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from configs import get_cfg_defaults

LOCAL_CONFIG_PATH = 'model/configs/dt_ultra.yaml'

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
        # depth always comes first
        if "depth" in dataset_output_type:
            output_names.append("depth")

        # add the other channels
        for d in ["x", "y", "z"]:
            for t in dataset_output_type:
                if t == "depth":
                    continue
                else:
                    # extend the three channels
                    output_names.append(f"{t}_{d}")

    else:
        for t in dataset_output_type:
            if t == "depth":
                output_names.append("depth")
            else:
                for d in ["x", "y", "z"]:
                    output_names.append(f"{t}_{d}")

    return output_names

def create_visualization(output_vis, output_names):
    """Create visualization grid of model outputs"""
    num_channels = output_vis.shape[0]
    cols = 5  # Show 5 columns
    rows = (num_channels + cols - 1) // cols  # Calculate rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i in range(num_channels):
        ax = axes[i]

        # Apply colormap based on output type
        if 'depth' in output_names[i]:
            im = ax.imshow(output_vis[i], cmap='viridis', vmin=0, vmax=output_vis[i].max())
        elif 'stress' in output_names[i] or 'disp' in output_names[i]:
            # Use diverging colormap for stress/displacement (can be positive/negative)
            vmax = max(abs(output_vis[i].min()), abs(output_vis[i].max()))
            im = ax.imshow(output_vis[i], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(output_vis[i], cmap='viridis')

        ax.set_title(f'{output_names[i]}\n[{output_vis[i].min():.2f}, {output_vis[i].max():.2f}]',
                     fontsize=8)
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Hide unused subplots
    for i in range(num_channels, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    # Convert figure to numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(canvas.get_width_height()[::-1] + (4,))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    plt.close(fig)
    return img

def main():
    print("Initializing TensorTouch Real-Time Tactile Sensor System...")

    # Load model
    print("Loading model from PyTorch Hub...")
    tactile_model = torch.hub.load('peasant98/DenseTact-Model', 'hiera',
                                    pretrained=True, map_location='cpu', trust_repo=True)
    tactile_model.eval()

    # Send model to cuda if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tactile_model = tactile_model.to(device)
    print(f"Model loaded on {device}")

    # Load config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(LOCAL_CONFIG_PATH)

    # Get output names
    output_names = get_output_names(cfg)
    print(f"Output channels: {output_names}")

    # Initialize camera
    print("\nInitializing camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("1. Position the tactile sensor in view of the camera")
    print("2. Press SPACE to capture the undeformed (reference) image")
    print("3. Apply pressure/deformation to the sensor")
    print("4. View real-time stress/displacement outputs")
    print("5. Press 'r' to recapture reference image")
    print("6. Press 'q' to quit")
    print("="*60 + "\n")

    undeformed_frame = None
    to_tensor = transforms.ToTensor()
    resize_transform = transforms.Resize((cfg.model.img_size, cfg.model.img_size), antialias=True)

    fps_counter = 0
    fps_start_time = cv2.getTickCount()
    current_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Calculate FPS
        fps_counter += 1
        if fps_counter >= 10:
            fps_end_time = cv2.getTickCount()
            time_diff = (fps_end_time - fps_start_time) / cv2.getTickFrequency()
            current_fps = fps_counter / time_diff
            fps_counter = 0
            fps_start_time = cv2.getTickCount()

        # Display current frame
        display_frame = frame.copy()

        # Add status text
        if undeformed_frame is None:
            cv2.putText(display_frame, "Press SPACE to capture reference image",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "Processing... FPS: {:.1f}".format(current_fps),
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Camera Feed', display_frame)

        key = cv2.waitKey(1) & 0xFF

        # Capture reference image
        if key == ord(' '):
            undeformed_frame = frame.copy()
            print("Reference image captured!")
            cv2.imwrite('model/reference_captured.png', undeformed_frame)
            continue

        # Recapture reference image
        if key == ord('r'):
            undeformed_frame = frame.copy()
            print("Reference image recaptured!")
            cv2.imwrite('model/reference_captured.png', undeformed_frame)
            continue

        # Quit
        if key == ord('q'):
            print("\nExiting...")
            break

        # Process if we have a reference frame
        if undeformed_frame is not None:
            try:
                # Convert frames to RGB
                deformed_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                undeformed_rgb = cv2.cvtColor(undeformed_frame, cv2.COLOR_BGR2RGB)

                # Convert to tensors and resize
                deformed_tensor = to_tensor(deformed_rgb)
                undeformed_tensor = to_tensor(undeformed_rgb)

                deformed_tensor = resize_transform(deformed_tensor)
                undeformed_tensor = resize_transform(undeformed_tensor)

                # Concatenate to create 6-channel input
                X = torch.cat([deformed_tensor, undeformed_tensor], dim=0)
                X = X.unsqueeze(0).to(device)  # Add batch dimension

                # Run inference
                with torch.no_grad():
                    output = tactile_model(X)
                    output_denorm = denormalize_output(output, cfg, output_names)

                # Visualize outputs
                output_vis = output_denorm[0].detach().cpu().numpy()
                vis_img = create_visualization(output_vis, output_names)

                cv2.imshow('Tactile Sensor Outputs', vis_img)

            except Exception as e:
                print(f"Error during processing: {e}")
                import traceback
                traceback.print_exc()

    cap.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == '__main__':
    main()
