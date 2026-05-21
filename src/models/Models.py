import torch.nn as nn
import torch
from .HigherModels import *
from efficientnet_pytorch import EfficientNet
import torchvision
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights, efficientnet_v2_m, EfficientNet_V2_M_Weights, efficientnet_v2_l, EfficientNet_V2_L_Weights
from math import floor, ceil

class ResNetAttention(nn.Module):
    def __init__(self, label_dim=527, pretrain=True, dropatt_rate = 0.2, dropout_rate=0.2):
        super(ResNetAttention, self).__init__()

        if pretrain == False:
            print('ResNet50 Model Trained from Scratch (ImageNet Pretraining NOT Used).')
            self.model = resnet50(weights=None)
        else:
            print('Now Use ImageNet Pretrained ResNet50 Model.')
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # remove the original ImageNet classification layers to save space.
        self.model.fc = torch.nn.Identity()
        self.model.avgpool = torch.nn.Identity()

        # attention pooling module
        self.attention = Attention(
            2048,
            label_dim,
            att_activation='sigmoid',
            cla_activation='sigmoid',
            dropatt_rate = dropatt_rate)
        self.avgpool = nn.AvgPool2d((4, 1))

        # Add dropout layer
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
    

    def forward(self, x):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        B, C, H, W = x.shape # [batch size, channel, freq_dim, time_dim]

        H_out = ceil(H / 32) # downsampling factor of ResNet-50
        W_out = ceil(W / 32)
        
        x = self.model(x) # [batch size, 2048 * H_out * W_out], e.g., [32, 131072] for input [32, 1, 128, 501]
        
        x = x.reshape(B, 2048, H_out, W_out) 
        x = self.avgpool(x)
        x = x.transpose(2,3)
        x = self.dropout(x)  # Apply dropout here
        out, norm_att = self.attention(x)
        return out

class MBNet(nn.Module):
    def __init__(self, label_dim=527, pretrain=True):
        super(MBNet, self).__init__()

        self.model = torchvision.models.mobilenet_v2(pretrained=pretrain)

        self.model.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier = torch.nn.Linear(in_features=1280, out_features=label_dim, bias=True)

    def forward(self, x, nframes):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        out = torch.sigmoid(self.model(x))
        return out


class EffNetAttention(nn.Module):
    def __init__(self, label_dim=527, b=0, pretrain=True, head_num=4, dropout_rate=0.2, dropatt_rate = 0.2, use_efficientnetv2=False, v2_model_name='s'):
        super(EffNetAttention, self).__init__()
        self.middim = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
        self.use_efficientnetv2 = use_efficientnetv2

        if self.use_efficientnetv2:
            if v2_model_name not in ['s', 'm', 'l']:
                raise ValueError("Invalid EfficientNetV2 model name. Choose from 's', 'm', or 'l'.")
            if v2_model_name == 's':
                print(f'Using EfficientNetV2-{v2_model_name} from torchvision')
                self.effnet = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            elif v2_model_name == 'm':
                print(f'Using EfficientNetV2-{v2_model_name} from torchvision')
                self.effnet = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
            elif v2_model_name == 'l':
                print(f'Using EfficientNetV2-{v2_model_name} from torchvision')
                self.effnet = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1) 
            self.effnet.classifier = nn.Identity()
        else:
            if pretrain == False:
                print('EfficientNet Model Trained from Scratch (ImageNet Pretraining NOT Used).')
                self.effnet = EfficientNet.from_name('efficientnet-b'+str(b), in_channels=1)
            else:
                print('Now Use ImageNet Pretrained EfficientNet-B{:d} Model.'.format(b))
                self.effnet = EfficientNet.from_pretrained('efficientnet-b'+str(b), in_channels=1)
        
        # multi-head attention pooling
        if head_num > 1:
            print('Model with {:d} attention heads'.format(head_num))
            self.attention = MHeadAttention(
                self.middim[b],
                label_dim,
                att_activation='sigmoid',
                cla_activation='sigmoid',
                dropatt_rate = dropatt_rate)
        # single-head attention pooling
        elif head_num == 1:
            print('Model with single attention heads')
            self.attention = Attention(
                self.middim[b],
                label_dim,
                att_activation='sigmoid',
                cla_activation='sigmoid')
        # mean pooling (no attention)
        elif head_num == 0:
            print('Model with mean pooling (NO Attention Heads)')
            self.attention = MeanPooling(
                self.middim[b],
                label_dim,
                att_activation='sigmoid',
                cla_activation='sigmoid')
        else:
            raise ValueError('Attention head must be integer >= 0, 0=mean pooling, 1=single-head attention, >1=multi-head attention.')

        self.avgpool = nn.AvgPool2d((4, 1))
        #remove the original ImageNet classification layers to save space.
        self.effnet._fc = nn.Identity()
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x, nframes=1056):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1).transpose(2, 3) # (16, 1, 128, 501)
        if self.use_efficientnetv2:
            x = x.repeat(1, 3, 1, 1)
            x = self.effnet.features(x)
        else:
            x = self.effnet.extract_features(x) # (16, middim[<b>], 4, 16)
        x = self.avgpool(x) # (16, middim[<b>], 4, 1)
        x = x.transpose(2,3) # (16, middim[<b>], 1, 4)
        x = self.dropout(x) # (16, middim[<b>], 1, 4)
        out, norm_att = self.attention(x) # (16, label_dim)
        return out