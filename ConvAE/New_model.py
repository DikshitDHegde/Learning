from Model import testContrastive
import math
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.modules.conv import Conv2d
from torchsummary import summary


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels, **kwargs)
        self.batch = nn.BatchNorm2d(out_channels)
        self.acti = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.acti(x)
        return x


class ConvEncoder(nn.Module):
    def __init__(self, in_channels, resolution):
        super().__init__()
        self.encoderConv1 = nn.Sequential(
            BasicConv2d(in_channels=in_channels, out_channels=32,
                        kernel_size=3, padding=1, stride=1),
            BasicConv2d(in_channels=32, out_channels=32,
                        kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(2)
        )

        self.encoderConv2 = nn.Sequential(
            BasicConv2d(in_channels=32, out_channels=64,
                        kernel_size=3, padding=1, stride=1),
            BasicConv2d(in_channels=64, out_channels=64,
                        kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(2)
        )

        self.encoderConv3 = nn.Sequential(
            BasicConv2d(in_channels=64, out_channels=256,
                        kernel_size=3, padding=1, stride=1),
            BasicConv2d(in_channels=256, out_channels=256,
                        kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(2)
        )

        self.width = 3

        self.fc = nn.Sequential(
            nn.Linear(self.width*self.width*256, 500),
            nn.ReLU(),
            nn.Linear(500, 256)
        )

    def forward(self, x):
        x = self.encoderConv1(x)
        x = self.encoderConv2(x)
        x = self.encoderConv3(x)

        x = x.view(x.shape[0], -1)

        x = self.fc(x)

        return x


class Projector(nn.Module):
    def __init__(self, input_):
        super().__init__()
        self.Proj = nn.Sequential(
            nn.Linear(input_, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, 10)
        )

    def forward(self, x):
        x = self.Proj(x)
        return x


class ConvDecoder(nn.Module):
    def __init__(self, out_channels, resolution):
        super().__init__()

        self.width = 3
        self.fc = nn.Sequential(
            nn.Linear(10, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, self.width*self.width*256),
            nn.ReLU(),
        )

        self.decoderConv3 = nn.Sequential(
            BasicConv2d(in_channels=256, out_channels=64,
                        kernel_size=3, stride=1, padding=2),
            BasicConv2d(in_channels=64, out_channels=64,
                        kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2)
        )

        self.decoderConv2 = nn.Sequential(
            BasicConv2d(in_channels=64, out_channels=32,
                        kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels=32, out_channels=32,
                        kernel_size=3, stride=1, padding=0),
            nn.Upsample(scale_factor=2)
        )

        self.decoderConv1 = nn.Sequential(
            BasicConv2d(in_channels=32, out_channels=16,
                        kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels=16, out_channels=out_channels,
                        kernel_size=3, stride=1, padding=0),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)

        x = x.view(x.shape[0], 256, self.width, self.width)
        # print(x.shape)

        x = self.decoderConv3(x)
        # print(x.shape)
        x = self.decoderConv2(x)
        # print(x.shape)
        x = self.decoderConv1(x)
        # print(x.shape)

        return x


class ConvContrastive(nn.Module):
    def __init__(self, in_channels, resolution):
        super().__init__()
        self.Encoder = ConvEncoder(
            in_channels=in_channels, resolution=resolution)
        self.Latent1 = nn.Linear(256, 256)
        self.Projector = Projector(256*2)
        self.Latent2 = nn.Linear(10, 10)
        self.Decoder = ConvDecoder(
            out_channels=in_channels, resolution=resolution)

    def forward(self, weak_x, strong_x):
        # print(weak_x.shape)
        weak_z = self.Encoder(weak_x)
        strong_z = self.Encoder(strong_x)

        weak_z = self.Latent1(weak_z)
        strong_z = self.Latent1(strong_z)

        features = torch.cat((weak_z, strong_z), dim=1)

        latent = self.Projector(features)

        latent = self.Latent2(latent)

        recon = self.Decoder(latent)
        # print(recon.shape)

        return weak_z, strong_z, latent, recon


class VAEConvContrastive(nn.Module):
    def __init__(self, in_channels, resolution):
        super().__init__()
        self.Encoder = ConvEncoder(
            in_channels=in_channels, resolution=resolution)
        self.mu1 = nn.Linear(256, 256)
        self.var1 = nn.Linear(256, 256)

        self.mu2 = nn.Linear(256, 256)
        self.var2 = nn.Linear(256, 256)
        self.Projector = Projector(256*2)
        self.Latent2 = nn.Linear(10, 10)
        self.Decoder = ConvDecoder(
            out_channels=in_channels, resolution=resolution)

    def re(self, mu, var):
        std = torch.exp(var*0.5)
        eps = torch.randn_like(std)
        return std*eps + mu

    def forward(self, weak_x, strong_x):
        weak_z = self.Encoder(weak_x)
        strong_z = self.Encoder(strong_x)

        weak_mu = self.mu1(weak_z)
        weak_var = self.var1(weak_z)
        strong_mu = self.mu2(strong_z)
        strong_var = self.var2(strong_z)

        weak_z = self.re(weak_mu, weak_var)
        strong_z = self.re(strong_mu, strong_var)

        features = torch.cat((weak_z, strong_z), dim=1)

        latent = self.Projector(features)

        latent = self.Latent2(latent)

        recon = self.Decoder(latent)
        # print(recon.shape)
        return weak_z, strong_z, latent, recon, weak_mu, weak_var, strong_mu, strong_var


def testContrastive():
    input1 = torch.rand((10,1,28,28))
    print(input1.shape)
    model = VAEConvContrastive(1,28)
    z1,z2,latent,recon,_,_,_,_ = model(input1,input1)
    reco = nn.MSELoss()
    eloss_sim =  nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')

    rloss = reco(recon,input1)
    rloss.backward(retain_graph=True)

    
    eloss = eloss_sim(z2,z1,torch.tensor(1))
    eloss.backward()


    print(eloss.item())
    print(model)

if __name__ == "__main__":
    testContrastive()