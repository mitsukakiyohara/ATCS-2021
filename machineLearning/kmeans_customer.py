"""
A K means clustering model to segmnet customers for a store 
"""

'''Import libraries'''
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans

'''Load data'''
data = pd.read_csv("data/customer.csv")
x = data[["Annual Income", "Spending Score"]]

# Standardize data
scaler = StandardScaler().fit(x)
x = scaler.transform(x)

'''Determine the value for K'''
# Calculate the intertia (sum of distances) from K=1 to 10
inertias = []
# Loop through K values
for k in range(1, 11): 
    # Build and fit the model 
    kmeanModel = KMeans(n_clusters=k).fit(x)
    # Store the inertias 
    inertias.append(kmeanModel.inertia_)

# Plot the inertias to find the Elbow 
# plt.plot(range(1,11), inertias, "bx-")
# plt.xlabel("Values of K")
# plt.ylabel("Inertia")
# plt.title("The Elbow Method using Inertia")
# plt.show() 
# # From graph: best K value = 5 

'''Create the Model'''
# From the elbow method: 
k = 5
km = KMeans(n_clusters=k).fit(x)

# Get the centroid and label values
centroid = km.cluster_centers_
labels = km.labels_
# labels in range [0, 4]

'''Visualize the clusters'''
# Works for 2D arrays 
# Set the size of the graph 
plt.figure(figsize=(5,4))

# Plot the data points for each of the k clusters 
for i in range(k):
    # Get all points x[n] where labels[n] == the label i 
    cluster = x[labels==i]
    # Get the income and spending values for each point in the cluster 
    cluster_income = cluster[:, 0] # First value from every row
    cluster_spending = cluster[:, 1] # Second value from every row
    
    plt.scatter(cluster_income, cluster_spending)

#Plot the centroids
centroids_income = centroid[:, 0]
centroids_spending = centroid[:, 1]
plt.scatter(centroids_income, centroids_spending, marker="x", color="red")

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()
