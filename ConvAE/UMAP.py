import pickle

import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap

with open('/home/beast/Experimentations/Dikshit/Contrastive Learning/CONV Learning/log/CONTRASTIVE/CIFAR_0009/FMNIST/Latent/Z1_190_.pickle', 'rb') as handle:
    data = pickle.load(handle)

z = data['latent']
target = data['target']

standard_embedding = umap.UMAP(random_state=42).fit(z.data)

for i in [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]:
    print(i)
    with open(f'/home/beast/Experimentations/Dikshit/Contrastive Learning/CONV Learning/log/CONTRASTIVE/CIFAR_0009/FMNIST/Latent/Z1_{i}_.pickle', 'rb') as handle:
        data = pickle.load(handle)

    z = data['latent']
    target = data['target']
    test_embedding = standard_embedding.transform(z.data)
    plt.figure()
    plt.scatter(test_embedding[:, 0], test_embedding[:, 1], c=target, s=0.1, cmap='Spectral')
    plt.savefig("CIFAR_%06d.png"%i)
