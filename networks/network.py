import timm
import torch
import logging
from torch import nn

from args import args


class Transformer(nn.Module):

    def __init__(self, scale='base', is_pretrained=True):

        super(Transformer, self).__init__()

        if scale == 'base':
            self.model = timm.create_model(model_name=f'vit_base_patch16_224', pretrained=is_pretrained)
        elif scale == 'large':
            self.model = timm.create_model(model_name=f'vit_large_patch16_224', pretrained=is_pretrained)
        else:
            logging.warning("scale is invalid!")

    def forward(self, input):

        feature = self.model(input)

        return feature


class ChannelAttention(nn.Module):

    def __init__(self, in_channels, reduction_ratio=2):
        
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):

        avg_pool = self.avg_pool(x).squeeze(2).squeeze(2)

        attention_map = self.fc(avg_pool).unsqueeze(2).unsqueeze(3)

        return x * attention_map


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):

        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)

        combined = torch.cat([avg_pool, max_pool], dim=1)
        attention_map = self.sigmoid(self.conv(combined))

        return x * attention_map


class Inception(nn.Module):

    def __init__(self, inchannel, outchannel):

        super(Inception, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=(1, 1), stride=1, padding=0),
            )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=(1, 1), stride=1, padding=0),
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=(3, 3), stride=1, padding=1),
            )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=(1, 1), stride=1, padding=0),
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=(5, 5), stride=1, padding=2),
            )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=(1, 1), stride=1, padding=0),
        )
    
    def forward(self, x):

        feature1 = self.branch1(x)
        feature2 = self.branch2(x)
        feature3 = self.branch3(x)
        feature4 = self.branch4(x)
        feature_c = torch.concat((feature1, feature2, feature3, feature4), dim=1)

        return feature_c


class predict_net(nn.Module):

    def __init__(self, inchannel, outchannel, transformer_scare='base',  is_pretrained=True):
        super(predict_net, self).__init__()

        self.model_transformer = Transformer(scale=transformer_scare, is_pretrained=is_pretrained)
        num_ftrs = self.model_transformer.model.head.in_features
        self.model_transformer.model.head = nn.Linear(num_ftrs, args.classes)

        self.inception = Inception(inchannel=inchannel, outchannel=outchannel)
        self.channel_attention = ChannelAttention(in_channels=outchannel*4)
        self.spatial_attention = SpatialAttention()

        self.conv_merge = nn.Sequential(
            nn.Conv2d(in_channels=outchannel*4, out_channels=3, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            )   

    def forward(self, x):

        feature_concat = self.inception(x)

        ca_feature = self.channel_attention(feature_concat)
        sa_feature = self.spatial_attention(ca_feature)

        feature = self.conv_merge(sa_feature) 
        
        output = self.model_transformer(feature)

        return output
