import torch
import torch.nn as nn



class DPTV2Net(nn.Module):
    def __init__(
        self, 
        img_size=256,
        patch_size=16,
        in_chans=3,
        encoder='vitl', 
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        out_dims=15,
        use_bn=False, 
        use_clstoken=False
    ):
        super(DPTV2Net, self).__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        self.encoder = encoder
        # use the dinov2 encoder from

        self.pretrained = DINOv2(model_name=encoder, img_size=img_size, in_chans=in_chans, patch_size=patch_size)
        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, output_dim=out_dims, 
                                out_channels=out_channels, use_clstoken=use_clstoken)
    
    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
        # extract intermediate features.
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        # predictions
        depth = self.depth_head(features, patch_h, patch_w)

        return depth
    
    def freeze_encoder(self):
        for param in self.pretrained.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.pretrained.parameters():
            param.requires_grad = True

    def load_from_pretrained_model(self, model_path:str):
        """
        Load weights from a pre-trained model
        """
        model_ckpt = torch.load(model_path)

        # load encoder weights
        self.pretrained.load_state_dict(model_ckpt["model"]['pretrained'])


if __name__ == '__main__':
    # dino vit + dpt head combo