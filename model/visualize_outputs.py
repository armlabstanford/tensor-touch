import torch
import argparse
from configs import get_cfg_defaults

import torchvision.models as models
import torch.nn.functional as F
from torchvision import transforms
from process_data import FullDataset


class ModelVisualizer():
    def __init__(self):
        self.model = None
        self.dataset = None
        
    def load_model(self, model):
        self.model = model
        
    def visualize_output(self, idx):
        """
        visualize outputs of the model
        
        """
        X, y = self.dataset[idx]

if __name__ == '__main__':
    # argparse for config file
    arg = argparse.ArgumentParser()
    arg.add_argument('--exp_name', type=str, default="exp/base")
    arg.add_argument('--ckpt_path', type=str, default=None)
    arg.add_argument('--config', type=str, default="configs/densenet_real_all.yaml")
    arg.add_argument('--eval', action='store_true')
    arg.add_argument('--real_world', action='store_true')
    opt = arg.parse_args()
    
    import pdb; pdb.set_trace()
    
    # get model path
    model_path = opt.ckpt_path
    cfg = get_cfg_defaults()
    cfg.merge_from_file(opt.config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((cfg.model.img_size, cfg.model.img_size), antialias=True),
    ])
    
    dataset = FullDataset(transform=transform, samples_dir=cfg.dataset.dataset_path,
                          output_type=cfg.dataset.output_type, is_real_world=opt.real_world)
    
    
    print("Dataset total samples: {}".format(len(dataset)))
    full_dataset_length = len(dataset)

    dataset_length = int(cfg.dataset_ratio * full_dataset_length)
    
    
    
    calibration_model = LightningDTModel(cfg)
    
    model_state = torch.load(opt.ckpt_path)
    calibration_model.model.decoders[0].head.conv1 = nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    del calibration_model.model.decoders[0].head.convs
    calibration_model.load_state_dict(model_state["state_dict"])
    
    # load the n heads
    decoder_N_head_info = {'heads': 9, 'channels_per_head': 1}
    head = DecoderNHead(64, **decoder_N_head_info)
    calibration_model.model.decoders[0].head = head
    opt.ckpt_path = None