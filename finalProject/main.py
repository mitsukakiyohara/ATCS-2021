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

''' Performing PCA 
PCA is a technique for feature extraction, where it combines input variables and drops "least important" variables. 
Gurantees that all the "new" variables are independent of each other. 
'''
# Credit: Paul J. Atzberger 

scalar = StandardScaler()

# Create a "design matrix" for the sample 
num_samples = data.shape[0]
image_size = data.shape[1] * data.shape[2] # 64 by 64
X = data.reshape((num_samples, image_size))



                        
        
    