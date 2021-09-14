import os
import pickle
import random

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import SSIM
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.utils.linear_assignment_ import linear_assignment
from torch.utils.data import Dataset
from tqdm import tqdm


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def save_latent(dir, epoch, latent, y, name):
    data = {"latent": latent, "target": y}
    with open(os.path.join(dir, f"{name}_{epoch}_.pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def Cluster(args, model, pretrain_loader, device, epoch):
    model.eval()
    Z = []
    y = []
    with torch.no_grad():
        loop = tqdm(enumerate(pretrain_loader),
                    total=len(pretrain_loader), leave=False)
        for idx, (weak_x, _, Y) in loop:
            weak_x = weak_x.to(device)
            y.extend(Y.data.cpu().numpy())
            _, _, z1, _, = model(weak_x, weak_x)
            Z.extend(z1.data.cpu().numpy())

        Z = np.array(Z)
        y = np.array(y)

        kmeans1 = KMeans(n_clusters=args.n_clusters, n_init=20, n_jobs=-1)
        y_pred1 = kmeans1.fit_predict(Z)

        kmeans1 = KMeans(n_clusters=args.n_clusters, n_init=20, n_jobs=-1)
        y_pred1 = kmeans1.fit_predict(z1.data.cpu().numpy())

        # print(y_pred1.shape,y.shape)

        nmi_1 = nmi_score(y_pred1, y)
        acc_1 = cluster_acc(y_pred1, y)
        ari_1 = ari_score(y_pred1, y)

        print(
            f"Epoch:[{epoch}/{args.epochs}] Latent space  : ACC:{acc_1:.4f} NMI:{nmi_1:.4f} ARI:{ari_1:.4f}")
        k = os.path.join(args.out_dir, "Latent")
        if not os.path.exists(k):
            os.makedirs(k)
        save_latent(k, epoch+1, z1.detach().cpu().numpy(), y, "Z1")

        return (acc_1, nmi_1, ari_1)


def VAECluster(args, model, pretrain_loader, device, epoch):
    model.eval()
    Z = []
    y = []
    with torch.no_grad():
        loop = tqdm(enumerate(pretrain_loader),
                    total=len(pretrain_loader), leave=False)
        for idx, (weak_x, _, Y) in loop:
            weak_x = weak_x.to(device)
            y.extend(Y.data.cpu().numpy())
            _, _, z1, _, _, _, _, _ = model(weak_x, weak_x)
            Z.extend(z1.data.cpu().numpy())

        Z = np.array(Z)
        y = np.array(y)
        kmeans1 = KMeans(n_clusters=args.n_clusters, n_init=20, n_jobs=-1)
        y_pred1 = kmeans1.fit_predict(Z)

        # print(y_pred1.shape,y.shape)

        nmi_1 = nmi_score(y_pred1, y)
        acc_1 = cluster_acc(y_pred1, y)
        ari_1 = ari_score(y_pred1, y)

        print(
            f"Epoch:[{epoch}/{args.epochs}] Latent space  : ACC:{acc_1:.4f} NMI:{nmi_1:.4f} ARI:{ari_1:.4f}")
        k = os.path.join(args.out_dir, "FMNIST", "Latent")
        if not os.path.exists(k):
            os.makedirs(k)
        save_latent(k, epoch+1, Z, y, "Z1")

        return (acc_1, nmi_1, ari_1)


def pretrain(args, model, pretrain_loader, device, optimizer, criterion_mse, criterion_emb, epoch):
    model.train()
    loop = tqdm(enumerate(pretrain_loader),
                total=len(pretrain_loader), leave=False)
    total_recon_loss = 0
    total_emb_loss = 0
    for idx, (weak_x, strong_x, _) in loop:
        weak_x = weak_x.to(device)
        strong_x = strong_x.to(device)
        # strong_x = strong_x.reshape(strong_x.shape[0],-1)
        # weak_x = weak_x.reshape(weak_x.shape[0],-1)
        optimizer.zero_grad()
        z1, z2, _, recon = model(weak_x, strong_x)

        # weak_x = reshape(weak_x,1,28)

        rloss = criterion_mse(recon, weak_x)
        rloss.backward(retain_graph=True)
        # for params in model.Encoder2.parameters():
        #     params.requires_grad = True

        # for params in model.latent1.parameters():
        #     params.requires_grad =False

        # for params in model.latent2.parameters():
        #     params.requires_grad =False

        emb_loss = criterion_emb(z1, z2)

        emb_loss.backward()

        total_emb_loss += emb_loss.item()*weak_x.shape[0]
        total_recon_loss += rloss.item()*weak_x.shape[0]
        optimizer.step()

        if idx % 10 == 0:
            loop.set_description(f"[{epoch}/{args.epochs}]:")

    return total_emb_loss/70000, total_recon_loss/70000


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -5e-4 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return torch.mean(0.5*BCE + KLD)


def reshape(x, channel, resolution):
    return x.reshape(x.shape[0], channel, resolution, resolution)


def pretrainVAE(args, model, pretrain_loader, device, optimizer, criterion_mse, criterion_emb, epoch):
    model.train()
    loop = tqdm(enumerate(pretrain_loader),
                total=len(pretrain_loader), leave=False)
    total_recon_loss = 0
    total_emb_loss = 0
    for idx, (weak_x, strong_x, _) in loop:
        weak_x = weak_x.to(device)
        strong_x = strong_x.to(device)
        # strong_x = strong_x.reshape(strong_x.shape[0],-1)
        # weak_x = weak_x.reshape(weak_x.shape[0],-1)
        optimizer.zero_grad()
        z1, z2, _, recon, weak_mu, weak_var, strong_mu, strong_var = model(
            weak_x, strong_x)

        # weak_x = reshape(weak_x,1,28)
        # strong_x = reshape(strong_x,1,28)
        rloss = criterion_mse(recon, weak_x)
        vloss1 = loss_function(recon, weak_x, weak_mu, weak_var)
        vloss2 = loss_function(recon, strong_x, strong_mu, strong_var)

        rloss += (vloss1 + vloss2)
        rloss.backward(retain_graph=True)
        emb_loss = criterion_emb(z1, z2)

        emb_loss.backward()

        total_emb_loss += emb_loss.item()*weak_x.shape[0]
        total_recon_loss += rloss.item()*weak_x.shape[0]
        optimizer.step()

        if idx % 10 == 0:
            loop.set_description(f"[{epoch}/{args.epochs}]:")

    return total_emb_loss/70000, total_recon_loss/70000


def set_seed_globally(seed_value=0, if_cuda=True):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if if_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = True


# def info_nce_loss(features,batch_size=256,device='cpu',temperature=0.01):

#         labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
#         labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
#         labels = labels.to(device)

#         features = F.normalize(features, dim=1)

#         similarity_matrix = torch.matmul(features, features.T)

#         mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
#         labels = labels[~mask].view(labels.shape[0], -1)
#         similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

#         # select and combine multiple positives
#         positives = 0.75*similarity_matrix[labels.bool()].view(labels.shape[0], -1)

#         # select only the negatives the negatives
#         negatives = 0.25*similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

#         logits = torch.cat([positives, negatives], dim=1).to(device)
#         labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
#         labels = labels.to(device)

#         logits = logits / temperature
#         return logits, labels
