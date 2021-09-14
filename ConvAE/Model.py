import math
from copy import deepcopy

import torch
import torch.nn as nn
from torchsummary import summary


class Encoder(nn.Module):
    def __init__(self,input_,layers=[1000,500,100,10]):
        super(Encoder, self).__init__()

        self.Enc = nn.ModuleList()

        for layer in layers:
            x = [nn.Linear(input_,layer),
                 nn.ReLU()]
            if layer == layers[-1]:
                x = [nn.Linear(input_,layer)]
            
            self.Enc.append(nn.Sequential(*x))

            input_= layer
    
    def forward(self, x):
        for ENC in self.Enc:
            x = ENC(x)
        
        return x

class Projector(nn.Module):
    def __init__(self,input_,layers=[256,10]):
        super().__init__()
        self.Proj = nn.ModuleList()

        for layer in layers:
            x = [
                nn.Linear(input_,layer),
                nn.ReLU()
            ]
            if layer == layers[-1]:
                x = [
                    nn.Linear(input_,layer)
                ] 
            
            self.Proj.append(nn.Sequential(*x))
            input_ = layer

    def forward(self,x):
        for PROJ in self.Proj:
            x = PROJ(x)
        
        return x

class Decoder(nn.Module):
    def __init__(self,input_,layers=[10,100,500,1000,784]):
        super(Decoder,self).__init__()

        self.Dec = nn.ModuleList()

        for layer in layers:
            x = [nn.Linear(input_,layer),
                 nn.ReLU()]
            if layer == layers[-1]:
                x = [nn.Linear(input_,layer),
                     nn.Sigmoid()]
            
            self.Dec.append(nn.Sequential(*x))

            input_= layer
        
    def forward(self, x):
        for DEC in self.Dec:
            x = DEC(x)
        
        return x
    

def testEnc():
    input_ = torch.randn((1,784))
    model = Encoder(
        input_=784,
        layers=[1000,500,100,10]
    )
    output = model(input_)
    summary(model,(1,784),device="cpu")
    print(model)
    print(output.shape)

def testDec():
    input_ = torch.randn((1,10))
    model = Decoder(
        input_=10,
        layers=[10,100,500,1000,784]
    )
    output = model(input_)
    summary(model,(1,10),device="cpu")
    print(output.shape)
    print(model)

def testProj():
    input_ = torch.randn((1,10))
    model = Projector(
        input_=10,
        layers=[10,100,500,1000,784]
    )
    output = model(input_)
    summary(model,(1,10),device="cpu")
    print(output.shape)
    print(model)

def copy_weights(shared_model,current_model):

    for current_params,shared_params in zip(current_model.parameters(),shared_model.parameters()):
        shared_params.data=current_params.data

class Contrastive(nn.Module):
    def __init__(self,input_,encoderLayers=[1000,500,100,10],projLayers=[256,128,10],decoderLayers=[10,2000,500,500,784],in_channels = 1, resolution=28):
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.Encoder1 = Encoder(
            input_=input_,
            layers=encoderLayers,
        )
        self.Encoder2 = self.Encoder1
        self.latent1 = nn.Linear(encoderLayers[-1],encoderLayers[-1])
        self.latent2 = self.latent1

        self.Projector1 = Projector(
            input_=2*encoderLayers[-1],
            layers=projLayers
        )
        self.latent3 = nn.Linear(projLayers[-1],projLayers[-1])

        self.Decoder = Decoder(
            input_=projLayers[-1],
            layers=decoderLayers
        )


    def re(self,mu,var):
        std = torch.exp(0.5*var)
        eps = torch.rand_like(std)
        return std*eps + mu

    def forward(self,weakx,strongx):
        weakx = weakx.reshape(weakx.shape[0],-1)
        strongx = weakx.reshape(strongx.shape[0],-1)
        weakx = self.Encoder1(weakx)
        weak_z = self.latent1(weakx)


        strongx = self.Encoder2(strongx)
        strong_z = self.latent2(strongx)

        Z = torch.cat((weak_z,strong_z),dim=1)

        latent = self.Projector1(Z)

        latent = self.latent3(latent)

        recon = self.Decoder(latent)
        recon = recon.reshape(recon.shape[0], self.in_channels, self.resolution, self.resolution)
        return weak_z,strong_z,latent,recon

class VaContrastive(nn.Module):
    def __init__(self,input_,encoderLayers=[1000,500,100,10],projLayers=[256,128,10],decoderLayers=[10,2000,500,500,784],in_channels = 1, resolution=28):
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.Encoder1 = Encoder(
            input_=input_,
            layers=encoderLayers,
        )
        self.Encoder2 = self.Encoder1
        self.mu1 = nn.Linear(encoderLayers[-1],encoderLayers[-1])
        self.mu2 = self.mu1

        self.var1 = nn.Linear(encoderLayers[-1],encoderLayers[-1])
        self.var2 = self.var1

        self.Projector1 = Projector(
            input_=2*encoderLayers[-1],
            layers=projLayers
        )
        self.latent3 = nn.Linear(projLayers[-1],projLayers[-1])

        self.Decoder = Decoder(
            input_=projLayers[-1],
            layers=decoderLayers
        )


    def re(self,mu,var):
        std = torch.exp(var*0.5)
        eps = torch.randn_like(std)
        return std*eps + mu

    def forward(self,weakx,strongx):
        weakx = weakx.reshape(weakx.shape[0],-1)
        strongx = weakx.reshape(strongx.shape[0],-1)
        weakx = self.Encoder1(weakx)
        weak_mu = self.mu1(weakx)
        weak_var = self.var1(weakx)

        weak_z = self.re(weak_mu,weak_var)

        strongx = self.Encoder2(strongx)
        strong_mu = self.mu2(strongx)
        strong_var = self.var2(strongx)

        strong_z = self.re(strong_mu,strong_var)

        Z = torch.cat((weak_z,strong_z),dim=1)

        latent = self.Projector1(Z)

        latent = self.latent3(latent)

        recon = self.Decoder(latent)
        recon = recon.reshape(recon.shape[0], self.in_channels, self.resolution, self.resolution)
        return weak_z,strong_z,latent,recon,weak_mu,weak_var,strong_mu,strong_var

def testContrastive():
    input1 = torch.randn((10,784))
    input2 = torch.randn((10,784))
    model = Contrastive(
        input_= 784,
        encoderLayers=[1000,500,128],
        projLayers=[256,100,10],
        decoderLayers=[10,100,500,1000,784]
    )

    z1,z2,latent,recon = model(input1,input2)

    for params in model.Encoder2.parameters():
        params.requires_grad =False
    
    for params in model.latent2.parameters():
        params.requires_grad =False
    
    reco = nn.MSELoss()
    eloss_sim =  nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')

    rloss = reco(recon,input1)
    rloss.backward(retain_graph=True)

    
    eloss = eloss_sim(z2,z1,torch.tensor(1))
    eloss.backward()


    print(eloss.item())
    print(model)


if __name__ == "__main__":
    # testEnc()
    # testDec()
    # testProj()
    testContrastive()
