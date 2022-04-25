"""
Facial Regonition Program with Olivetti Dataset
Name: Mitsuka Kiyohara
Date: 4/21/22 
Block: B 

"""

''' Import Libraries '''
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt 

''' Load Data '''
data = np.load("olivetti_faces.npy")
target_labels = np.load("olivetti_faces_target.npy") 

''' Performing PCA '''
# Transform from matrix to vector form 
num_samples = data.shape[0]
image_size = data.shape[1] * data.shape[2] # 64 by 64
X = data.reshape((num_samples, image_size))

# print("X.shape = " + str(X.shape))

# Find optimum number of principle component (n_component)
pca = PCA()
pca.fit(X)
plt.figure(1, figsize=(12,8))
plt.plot(pca.explained_variance_, linewidth=2)

plt.xlabel('Components')
plt.ylabel('Explained Variances')
plt.title('The Elbow Method Showing the Optimal Component')
plt.show()

''' Split data and target into random test and train subsets '''
# We will be splitting it 9:1 (90% for training, 10% for testing)

X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.1, stratify=target, random_state=0)

n_components = 90 
pca = PCA(n_components=n_components, whiten=True)
pca.fit(X_train)

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
axs.imshow(pca.mean_.reshape((64, 64)), cmap="gray")
axs.set_xticks([])
axs.set_yticks([])
axs.set_title('Average Face')



                        
        
    