import torch
import torch.nn as nn
import torchvision.models as models

class DTDenseNet(nn.Module):
    def __init__(self, out_chans=[3, 3, 3, 1]):
        super(DTDenseNet, self).__init__()
        
        # Encoder: Using a pre-trained DenseNet-161 model
        self.encoder = models.densenet161(pretrained=True)
        self.encoder.features.conv0 = nn.Conv2d(7, 96, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Extract the number of features from the DenseNet model
        self.encoder_out_channels = 2208  # For densenet161
        
        self.decoders = nn.ModuleList()
        for i, out_chan in enumerate(out_chans):
            decoder = nn.Sequential(
                nn.ConvTranspose2d(self.encoder_out_channels, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(True),
                nn.ConvTranspose2d(64, out_chan, kernel_size=4, stride=2, padding=1),
            )

            self.decoders.append(decoder)

    def forward(self, x):
        # Encoding
        features = self.encoder.features(x)
        
        # Reshape for decoder
        # x = features.view(features.size(0), self.encoder_out_channels, 1, 1)
        outputs = []

        # Decoding for each head
        for decoder in self.decoders:
            out = decoder(features)
            outputs.append(out)
        
        return torch.cat(outputs, dim=1)
    
    def load_from_pretrained_model(self, model_path:str):
        """
        Load weights from a pre-trained model
        """
        # print in red color

    def freeze_encoder(self):
        """
        Freeze the encoder weights
        """
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """
        Unfreeze the encoder weights
        """
        for param in self.encoder.parameters():
            param.requires_grad = True

class DecoderHead(nn.Module):
    """Decoder head module."""
    def __init__(self, in_chans=64, output_channels=1):
        super(DecoderHead, self).__init__()
        self.conv1 = nn.Conv2d(in_chans, output_channels, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.conv1(x)
        return x    

class DecoderNHead(nn.Module):
    """Decoder head module for n heads."""
    def __init__(self, in_chans=64, heads=5, channels_per_head=3):
        super(DecoderNHead, self).__init__()
        
        # Calculate the output channels based on the number of heads and channels per head
        self.output_channels = heads * channels_per_head
        
        # Create multiple convolutions for each head
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_chans, 32, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(16, channels_per_head, kernel_size=3, padding=1)
            ) 
            for _ in range(heads)
        ])
        
    def forward(self, x):
        # Apply each convolution and concatenate their outputs along the channel dimension
        
        outputs = [conv(x) for conv in self.convs]
        return torch.cat(outputs, dim=1)
    
    
class DecoderBlock(nn.Module):
    """Decoder block module using Upsampling and Convolution."""
    def __init__(self, in_channels, mid_channels, out_channels=None):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # use ct2d instead of convtranspose2d
        # self.upsample = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=2, stride=2)
        # if out_channels is None:
        #     self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        self.conv2 = None
        if out_channels is not None:
            self.conv2= nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True)
            )

        self.apply(self.init_weights)
    
    @torch.no_grad()
    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, self._interpolate(skip, x)], dim=1)
        x = self.conv1(x)
        
        if self.conv2 is not None:
            x = self.conv2(x)
            
        return x
        
    def _interpolate(self, enc_ftrs, x):
        """Resize the encoder features to match the dimensions of the decoder features."""
        return torch.nn.functional.interpolate(enc_ftrs, size=x.shape[2:], mode='bilinear', align_corners=False)
        
class ResizeConv(nn.Module):
    def __init__(self, in_channels, decoder_channels):
        super(ResizeConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, decoder_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.conv(x)
        return x
    
class FullDecoder(nn.Module):
    def __init__(self, encoder_num_features, decoder_features,
                        decoder_mid_dim, decoder_output_dim,
                        output_channels=1, decoder_N_head_info={'heads': 5, 'channels_per_head': 3}):
        """
        Full Decoder of DenseNet
        encoder_num_features List[int]: number of features from the encoder, from deep to shallow 
        decoder_features int: number of features in the first layer of the decoder
        decoder_mid_dim List[int]: number of features in the middle of the decoder
        decoder_output_dim List[int]: number of features in the output of the decoder
        output_channels int: number of output channels
        """
        super(FullDecoder, self).__init__()
        self.conv_resize = ResizeConv(encoder_num_features[0], decoder_features)
        
        self.decoder = nn.ModuleList()
        for i in range(1, len(encoder_num_features)):
            if i == 1:
                decoder_input_dim = decoder_features + encoder_num_features[i]
            else:
                decoder_input_dim = decoder_output_dim[i - 2] + encoder_num_features[i]
            self.decoder.append(DecoderBlock(decoder_input_dim,
                                             decoder_mid_dim[i - 1],
                                             decoder_output_dim[i - 1]))
        
        # no concatenation in the last layer
        # self.decoder.append(DecoderBlock(decoder_output_dim[-2], decoder_mid_dim[-1], decoder_output_dim[-1]))
        # [Deprecated] Use the following line instead of the above line for old weights
        self.decoder.append(DecoderBlock(decoder_output_dim[-2], decoder_mid_dim[-1])) # , decoder_output_dim[-1]))
        
        # use decoder 
        if decoder_N_head_info is not None:
            self.head = DecoderNHead(decoder_output_dim[-1], **decoder_N_head_info)
        else:
            self.head = DecoderHead(decoder_output_dim[-1], output_channels)
        
    def forward(self, x, skip_connections):
        x = self.conv_resize(x)
        
        for i, block in enumerate(self.decoder):
            x = block(x, skip=skip_connections[i])
            
        x = self.head(x)
        
        return x
    
class ResnetEncoder(nn.Module):
    def __init__(self, cfg):
        super(ResnetEncoder, self).__init__()
        self.encoder = getattr(models, cfg.model.backbone)(pretrained=cfg.model.imagenet_pretrained)
        last_feat_dim = self.encoder.fc.in_features

        self.encoder.conv1 = nn.Conv2d(cfg.model.in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        self.encoder[0] = nn.Conv2d(cfg.model.in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.feature_channels = [ last_feat_dim,
                                self.encoder[7][0].conv1.in_channels,
                                self.encoder[6][0].conv1.in_channels, 
                                self.encoder[5][0].conv1.in_channels,
                                self.encoder[4][0].conv1.in_channels]

    def forward(self, x):
        x1 = self.encoder[0](x)  # conv1
        x1 = self.encoder[1](x1)  # bn1
        x1 = self.encoder[2](x1)  # relu
        x1 = self.encoder[3](x1)  # maxpool
        x2 = self.encoder[4](x1)  # layer1
        x3 = self.encoder[5](x2)  # layer2
        x4 = self.encoder[6](x3)  # layer3
        x5 = self.encoder[7](x4)  # layer4
        
        return x1, x2, x3, x4, x5
    
    def get_feature_channels(self):
        """ get feature channels from the encoder, from deep to shallow """
        return self.feature_channels
    
class DensenetEncoder(nn.Module):
    def __init__(self, cfg):
        super(DensenetEncoder, self).__init__()
        self.encoder = getattr(models, cfg.model.backbone)(pretrained=cfg.model.imagenet_pretrained)
        self.feature_channels = [ self.encoder.classifier.in_features,
                                self.encoder.features.denseblock4.denselayer1.norm1.num_features,
                                self.encoder.features.denseblock3.denselayer1.norm1.num_features, 
                                self.encoder.features.denseblock2.denselayer1.norm1.num_features,
                                self.encoder.features.denseblock1.denselayer1.norm1.num_features]

        self.in_chans = cfg.model.in_chans
        # modify the first layer to accept 7 channels
        out_channels = self.encoder.features.conv0.out_channels
        kernel_size = self.encoder.features.conv0.kernel_size
        stride = self.encoder.features.conv0.stride
        padding = self.encoder.features.conv0.padding
        have_bias = self.encoder.features.conv0.bias is not None
        self.encoder.features.conv0 = nn.Conv2d(cfg.model.in_chans, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=have_bias)

    def forward(self, x):
        x1 = self.encoder.features.relu0(self.encoder.features.norm0(self.encoder.features.conv0(x)))
        x1 = self.encoder.features.pool0(x1)
        x2 = self.encoder.features.denseblock1(x1)
        x2 = self.encoder.features.transition1(x2)
        x3 = self.encoder.features.denseblock2(x2)
        x3 = self.encoder.features.transition2(x3)
        x4 = self.encoder.features.denseblock3(x3)
        x4 = self.encoder.features.transition3(x4)
        x5 = self.encoder.features.denseblock4(x4)

        return x1, x2, x3, x4, x5
    
    def get_feature_channels(self):
        """ get feature channels from the encoder, from deep to shallow """
        return self.feature_channels

class ResnextEncoder(nn.Module):
    def __init__(self, cfg):
        super(ResnextEncoder, self).__init__()
        backbone_name = cfg.model.backbone  # e.g., 'resnext101_32x8d'
        in_chans = cfg.model.in_chans       # Number of input channels
        pretrained = cfg.model.imagenet_pretrained  # Boolean for pretrained weights
        # Initialize the ResNeXt model
        self.encoder = getattr(models, backbone_name)(pretrained=pretrained)
        last_feat_dim = self.encoder.fc.in_features  # Feature dimension before the fully connected layer

        # Modify the first convolutional layer if input channels are not equal to 3
        if in_chans != 3:
            self.encoder.conv1 = nn.Conv2d(
                in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Remove the fully connected layer and the average pooling layer
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])

        # Extract feature channels from different layers
        self.feature_channels = [
            last_feat_dim,  # Output channels from layer4
            self.encoder[6][-1].conv3.out_channels,  # Output channels from layer2
            self.encoder[5][-1].conv3.out_channels,  # Output channels from layer1
            self.encoder[4][-1].conv3.out_channels,  # Output channels from layer1
            self.encoder[1].num_features,            # Output channels after maxpool
        ]

    def forward(self, x):
        
        # Initial convolution and pooling layers
        x = self.encoder[0](x)  # conv1
        x1 = self.encoder[1](x)  # bn1
        x = self.encoder[2](x1)  # relu
        x1 = self.encoder[3](x1)  # maxpool

        # Layer 1
        x2 = self.encoder[4](x1)

        # Layer 2
        x3 = self.encoder[5](x2)

        # Layer 3
        x4 = self.encoder[6](x3)

        # Layer 4
        x5 = self.encoder[7](x4)

        # Return features in reverse order (from deepest to shallowest)
        return x1, x2, x3, x4, x5
    
    def get_feature_channels(self):
        """ get feature channels from the encoder, from deep to shallow """
        return self.feature_channels

class DTNet(nn.Module):
    def __init__(self, cfg):
        super(DTNet, self).__init__()
        
        # Encoder: Using a pre-trained DenseNet-161 model
        encoder_features = None
        if cfg.model.encoder == 'densenet':
            self.encoder = DensenetEncoder(cfg)
            # remove classifier layer
            # [Deprecated] comment the following line for old weights
            del self.encoder.encoder.classifier
        # resnet series
        elif cfg.model.encoder == 'resnet':
            self.encoder = ResnetEncoder(cfg)
        elif cfg.model.encoder == 'resnext':
            self.encoder = ResnextEncoder(cfg)
        else:
            raise NotImplementedError("Encoder not implemented {}".format(cfg.model.encoder))
        encoder_features = self.encoder.get_feature_channels()
        
        self.decoders = nn.ModuleList()
        # start with 1024 features in decoder
        out_chans = [cfg.model.out_chans[0]]

        self.decoder_features = 1024
        for head_output_channels in out_chans:
            # Decoder: Using a custom decoder
            self.decoders.append(FullDecoder(encoder_features, self.decoder_features, 
                                            cfg.model.cnn.decoder_mid_dim, cfg.model.cnn.decoder_output_dim,
                                            output_channels=head_output_channels))
    
    def unfreeze_encoder(self):
        """
        Unfreeze the encoder weights
        """
        for param in self.encoder.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        # Densenet Encoder
        x1, x2, x3, x4, x5 = self.encoder(x)

        # for student teacher training, we will supervise the encoder output z
        z = x5
        
        # if decoders are none, return the encoder output
        if getattr(self, 'decoders', None) is None:
            return z
        
        # Decoder
        # go through each decoder
        outputs = []
        for decoder in self.decoders:
            outputs.append(decoder(x5, [x4, x3, x2, x1, None]))
        
        # combine the outputs into:
        # (B, head_output_channels x n_heads output, H, W)
        outputs = torch.cat(outputs, dim=1)
        
        return outputs, z
    
    def load_from_pretrained_model(self, model_path:str):
        model_state = torch.load(model_path)

        state_dict = {}
        for k in self.encoder.state_dict().keys():
            if 'model.encoder.' + k in model_state["state_dict"]:
                state_dict[k] = model_state["state_dict"]['model.encoder.' + k]

        # calibration_model_state_dict = self.encoder.state_dict()
        # calibration_model_state_dict.update(encoder_state_dict)
        self.encoder.load_state_dict(state_dict)

        decoder_state_dict = {}
        for k in self.decoders.state_dict().keys():
            if 'model.decoders.' + k in model_state["state_dict"]:
                decoder_state_dict[k] = model_state["state_dict"]['model.decoders.' + k]

        self.decoders.load_state_dict(decoder_state_dict)

        print(f"Loaded model from {model_path}")
    
if __name__ == "__main__":
    # Instantiate the model
    # sample usage with 1 heads and 1 output channels per head.
    
    class Config:
        class Model:
            encoder = 'resnext'
            backbone = 'resnext101_32x8d'
            imagenet_pretrained = True
            in_chans = 7  # Change this if you have a different number of input channels
            out_chans = 1
            
        class CNN:
            decoder_mid_dim = [1024, 1024, 512, 256, 64]
            decoder_output_dim = [1024, 512, 256, 128, 64]
        model = Model()
        model.cnn = CNN()
    cfg = Config()
    
    model = DTNet(cfg)
    
    
    # test input and output shape is desired
    dummy_input = torch.randn(1, 7, 256, 256)

    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_input)
        
    print(outputs.shape)
