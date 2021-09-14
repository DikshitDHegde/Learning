import math
from copy import deepcopy

import torch
import torch.nn as nn
from torchsummary import summary


class Encoder(nn.Module):
    def __init__(self,input_,layers=[1000,500,100,10]):
        super(Encoder, self).__init__()

        X = []

        for layer in layers:
            x = [nn.Linear(input_,layer),
                 nn.ReLU()]
            if layer == layers[-1]:
                x = [nn.Linear(input_,layer)]
            
            X.extend(x)
            input_= layer
        self.Enc = nn.Sequential(*X)

    def forward(self, x):
        x = self.Enc(x)        
        return x

class Projector(nn.Module):
    def __init__(self,input_,layers=[256,10]):
        super().__init__()
        X = []

        for layer in layers:
            x = [
                nn.Linear(input_,layer),
                nn.ReLU()
            ]
            if layer == layers[-1]:
                x = [
                    nn.Linear(input_,layer)
                ]
            
            X.extend(x)
            input_ = layer
        self.Proj = nn.Sequential(*X)

    def forward(self,x):
        x = self.Proj(x)
        return x

class Decoder(nn.Module):
    def __init__(self,input_,layers=[10,100,500,1000,784]):
        super(Decoder,self).__init__()

        X = []

        for layer in layers:
            x = [nn.Linear(input_,layer),
                 nn.ReLU()]
            if layer == layers[-1]:
                x = [nn.Linear(input_,layer),
                     nn.Sigmoid()]
            
            X.extend(x)
            input_= layer
        self.Dec = nn.Sequential(*X)
        
    def forward(self, x):
        x = self.Dec(x)
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

# class Autoencoder(nn.Module):
#     def __init__(self,input_,encoderLayers=[1000,500,100,10],decoderLayers=[10,100,500,1000,784]):
#         super().__init__()
#         self.Encoder1 = Encoder(
#             input_=input_,
#             layers=encoderLayers
#         )
#         self.latent1 = nn.Linear(encoderLayers[-1],encoderLayers[-1])

#         self.Decoder1 = Decoder(
#             input_=encoderLayers[-1],
#             layers=decoderLayers
#         )
    
#     def forward(self, x):
#         x = self.Encoder1(x)
#         x = self.latent1(x)
#         x = self.Decoder1(x)
#         return x


class Contrastive(nn.Module):
    def __init__(self,input_,encoderLayers=[1000,500,100,10],projLayers=[256,128,10],decoderLayers=[10,2000,500,500,784]):
        super().__init__()
        self.Encoder1 = Encoder(
            input_=input_,
            layers=encoderLayers,
        )
        self.latent1 = nn.Linear(encoderLayers[-1],encoderLayers[-1])

        self.Projector1 = Projector(
            input_=2*encoderLayers[-1],
            layers=projLayers
        )
        self.latent3 = nn.Linear(projLayers[-1],projLayers[-1])

        self.Decoder = Decoder(
            input_=projLayers[-1],
            layers=decoderLayers
        )

    def forward(self,weakx,strongx):
        weakx = self.Encoder1(weakx)
        weak_z = self.latent1(weakx)

        strongx = self.Encoder1(strongx)
        strong_z = self.latent1(strongx)
        Z = torch.cat((weak_z,strong_z),dim=1)
        latent = self.Projector1(Z)

        latent = self.latent3(latent)

        recon = self.Decoder(latent)

        return weak_z,strong_z,latent,recon

class VaContrastive(nn.Module):
    def __init__(self,input_,encoderLayers=[1000,500,100,10],projLayers=[256,128,10],decoderLayers=[10,2000,500,500,784]):
        super().__init__()
        self.Encoder1 = Encoder(
            input_=input_,
            layers=encoderLayers,
        )
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
        weakx = self.Encoder1(weakx)
        weak_mu = self.mu1(weakx)
        weak_var = self.var1(weakx)

        weak_z = self.re(weak_mu,weak_var)


        strongx = self.Encoder1(strongx)
        strong_mu = self.mu2(strongx)
        strong_var = self.var2(strongx)

        strong_z = self.re(strong_mu,strong_var)

        Z = torch.cat((weak_z,strong_z),dim=1)
        latent = self.Projector1(Z)

        latent = self.latent3(latent)

        recon = self.Decoder(latent)

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

    # summary(model,[(10,784),(10,784)],device="cpu")
    z1,z2,latent,recon = model(input1,input2)

    # for params in model.Encoder2.parameters():
    #     params.requires_grad =False
    
    # for params in model.latent2.parameters():
    #     params.requires_grad =False
    
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
