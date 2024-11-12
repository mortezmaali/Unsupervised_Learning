# -*- coding: utf-8 -*-
"""
Created on Fri May 31 20:58:42 2024

@author: Morteza
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import spectral as spy
import matplotlib.patches as mpatches

def hypermnf(cube, numComponents, mean_centered=True):
    h, w, channels = cube.shape
    cube = cube.reshape(h * w, channels)
    
    if mean_centered:
        u = np.mean(cube, axis=0)
        cube -= u

    V = np.diff(cube, axis=0)
    V = np.cov(V, rowvar=False)
    
    # Perform PCA for dimensionality reduction
    pca = PCA(n_components=numComponents, svd_solver='full')
    reduced_cube = pca.fit_transform(cube)
    
    return reduced_cube.reshape(h, w, numComponents)

# Main script execution
datafile = 'C:/Users/Morteza/OneDrive/Desktop/PhD/New_Data/8cal_Seurat_AFTER'
hdrfile = 'C:/Users/Morteza/OneDrive/Desktop/PhD/New_Data/8cal_Seurat_AFTER.hdr'

# Load the hyperspectral image using spectral library
hcube = spy.open_image(hdrfile)
img = hcube.load().astype(np.float64)

# Exclude the last band if it contains NaN values
if np.isnan(img[:, :, -1]).any():
    img = img[:, :, :-1]

# Preprocess the image (e.g., using MNF/PCA for dimensionality reduction)
img = img[:, :, :151]
numComponents = 10  # Reduce to 10 components for clustering
reduced_img = hypermnf(img, numComponents)

# Reshape image for clustering
h, w, _ = reduced_img.shape
img_reshaped = reduced_img.reshape(-1, numComponents)

# Apply K-means clustering
num_clusters = 9
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans_labels = kmeans.fit_predict(img_reshaped)

# Apply Gaussian Mixture Model for clustering
gmm = GaussianMixture(n_components=num_clusters, covariance_type='full', random_state=0)
gmm_labels = gmm.fit_predict(img_reshaped)

# Reshape the cluster labels to the original image shape
kmeans_image = kmeans_labels.reshape(h, w)
gmm_image = gmm_labels.reshape(h, w)

# Define class names (optional)
classNames = [f'class {i+1}' for i in range(num_clusters)]

# Plot the results for comparison
fig, ax = plt.subplots(1, 2, figsize=(15, 7))

# K-means result
ax[0].imshow(kmeans_image, cmap='tab10')
ax[0].set_title('K-means Clustering')
ax[0].set_aspect('equal', adjustable='box')
ax[0].set_xticks([])
ax[0].set_yticks([])

# GMM result
ax[1].imshow(gmm_image, cmap='tab10')
ax[1].set_title('Gaussian Mixture Model Clustering')
ax[1].set_aspect('equal', adjustable='box')
ax[1].set_xticks([])
ax[1].set_yticks([])

# Create dummy patches for legend
legend_patches = [mpatches.Patch(color=plt.cm.tab10(i), label=classNames[i]) for i in range(num_clusters)]

# Display legend
fig.legend(handles=legend_patches, loc='center right', bbox_to_anchor=(1.15, 0.5))

plt.show()
