import pickle

import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap

with open('/home/cvg-ws2/Desktop/Dikshit/Contrastive/Auto_VAE/FMNIST/Latent/Z1_110_.pickle', 'rb') as handle:
    data = pickle.load(handle)

z = data['latent']
target = data['target']

standard_embedding = umap.UMAP(random_state=42).fit_transform(z.data)

plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=target, s=0.1, cmap='Spectral')
plt.show()
