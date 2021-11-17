import torch.nn as nn
import torch
from .HigherModels import *
from efficientnet_pytorch import EfficientNet
import torchvision

class ResNetAttention(nn.Module):
    def __init__(self, label_dim=527, pretrain=True):
        super(ResNetAttention, self).__init__()

        self.model = torchvision.models.resnet50(pretrained=False)

        if pretrain == False:
            print('ResNet50 Model Trained from Scratch (ImageNet Pretraining NOT Used).')
        else:
            print('Now Use ImageNet Pretrained ResNet50 Model.')

        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # remove the original ImageNet classification layers to save space.
        self.model.fc = torch.nn.Identity()
        self.model.avgpool = torch.nn.Identity()

        # attention pooling module
        self.attention = Attention(
            2048,
            label_dim,
            att_activation='sigmoid',
            cla_activation='sigmoid')
        self.avgpool = nn.AvgPool2d((4, 1))

    def forward(self, x):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        batch_size = x.shape[0]
        x = self.model(x)
        x = x.reshape([batch_size, 2048, 4, 33])
        x = self.avgpool(x)
        x = x.transpose(2,3)
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
    def __init__(self, label_dim=527, b=0, pretrain=True, head_num=4):
        super(EffNetAttention, self).__init__()
        self.middim = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
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
                cla_activation='sigmoid')
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

    def forward(self, x, nframes=1056):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        x = self.effnet.extract_features(x)
        x = self.avgpool(x)
        x = x.transpose(2,3)
        out, norm_att = self.attention(x)
        return out

if __name__ == '__main__':
    input_tdim = 1056
    #ast_mdl = ResNetNewFullAttention(pretrain=False)
    psla_mdl = EffNetFullAttention(pretrain=False, b=0, head_num=0)
    # input a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins
    test_input = torch.rand([10, input_tdim, 128])
    test_output = psla_mdl(test_input)
    # output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes.
    print(test_output.shape)