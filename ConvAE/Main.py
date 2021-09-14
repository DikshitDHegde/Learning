import warnings

warnings.simplefilter(action='ignore') #, category=FutureWarning)
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchvision.utils import save_image
from tqdm import tqdm

import utils
import loss
from config import config
from Dataloader import RandAugmentCOLOR, RandAugmentMNIST, dataSetFn, loadData
# from New_model import VAEConvContrastive, ConvContrastive # Contrastive, VaContrastive
# from Cifar_Model import VAEConvContrastive, ConvContrastive # Contrastive, VaContrastive
from MINI_model import VAEConvContrastive, ConvContrastive # Contrastive, VaContrastive

# from pytorch_msssim import SSIM

torch.set_printoptions(linewidth=50)


def save_latent(dir, epoch, latent, y,name):
    data = {"latent": latent, "target": y}
    with open(os.path.join(dir, f"{name}_{epoch}_.pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_optim(optimze, model, learning_rate):
    if optimze == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimze == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        print("OPTIMIZER NOT IMPLEMENTED: SORRY")
        exit()

    return optimizer


use_gpu = torch.cuda.is_available()
# use_gpu = False
if use_gpu:
    device = "cuda"
    print(" USING CUDA :)")
else:
    device = "cpu"


batchsize = 1024
optimizer = "Adam"
# optimizer = "SGD"
lr = 1e-3
dataset = loadData("/home/beast/DATA/DATASET_NPZ/MINI_Combined.npz")
trainSet = dataSetFn(dataset=dataset, transform_original=transforms.ToTensor(), transform_weak=RandAugmentCOLOR(resolution=84,trans=transforms.ToTensor()))
trainLoader = DataLoader(trainSet, batch_size=batchsize, shuffle=True,
                        num_workers=10, pin_memory=False, prefetch_factor=batchsize//4)

for alpha in [0]:
    args = config()
    utils.set_seed_globally(0, use_gpu)

    # model = VaContrastive(
    #     input_= 784,
    #     encoderLayers=[500,500,2000,128],
    #     projLayers=[256,10],
    #     decoderLayers=[10,2000,500,500,784]
    # ).to(device)

    model  = VAEConvContrastive(in_channels=3,resolution=84).to(device)
    
    # print("DATALOADER")
    optimize = get_optim(optimizer, model, lr)
    # criterion_con = nn.CrossEntropyLoss().to(device)
    # criterion_mse = nn.MSELoss().to(device)
    criterion_mse = loss.reconLoss(in_channels=3, use_ssim=True, alpha=alpha).to(device)
    criterion_emb = loss.NCELoss(device=device)

    tb = SummaryWriter(
        f"./log/CONTRASTIVE/CIFAR/BatchSize {batchsize} LR {lr} Optimizer {optimizer} Alpha{alpha}")

    print(f"./log/CONTRASTIVE/CIFAR/BatchSize {batchsize} LR {lr} Optimizer {optimizer} Alpha{alpha}")

    for epoch in range(args.epochs):
        # emb_loss,recon_loss = utils.pretrain(args, model, trainLoader, device, optimize, criterion_mse, criterion_emb, epoch)
        emb_loss,recon_loss = utils.pretrainVAE(args, model, trainLoader, device, optimize, criterion_mse,criterion_emb, epoch)
        tb.add_scalar("Emb loss", emb_loss, global_step=epoch)
        tb.add_scalar("Recon loss", recon_loss, global_step=epoch)
        if (epoch+1) % 10 == 0:
            # acc_1, nmi_1, ari_1 = utils.Cluster(
            #     args, model, trainLoader, device, epoch)
            acc_1, nmi_1, ari_1 = utils.VAECluster(
                args, model, trainLoader, device, epoch)
            tb.add_scalar("Pretrain ACC L1", acc_1, global_step=epoch)

            tb.add_scalar("Pretrain NMI L1", nmi_1, global_step=epoch)

            tb.add_scalar("Pretrain ARI L1", ari_1, global_step=epoch)

        if (epoch+1) % 100 == 0:
            k = os.path.join(args.out_dir, "MNIST")
            if not os.path.exists(k):
                os.makedirs(k)

            torch.save(
                {
                    "weights": model.state_dict(),
                    "optimizer": optimize.state_dict(),
                    "epoch": epoch+1,
                    "Emb_loss": emb_loss
                }, os.path.join(k, f"BatchSize {batchsize} LR {lr}  Optim {optimizer} alpha{alpha}.pth.tar")
            )

    del model