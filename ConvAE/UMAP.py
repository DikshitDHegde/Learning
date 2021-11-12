import pickle

import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap

with open('/home/dikshit/Desktop/Dikshit/Latent/Z1_10_.pickle', 'rb') as handle:
    data = pickle.load(handle)

z = data['latent']
target = data['target']

standard_embedding = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.2, n_neighbors=15, min_dist=0.2).fit_transform(z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(standard_embedding[:,0],standard_embedding[:,1],standard_embedding[:,2], c=target, s=2, cmap='Spectral')
ax.axis("off")
# plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=target, s=0.1, cmap='Spectral')
plt.show()

n_neighbors=5, min_dist=0.1