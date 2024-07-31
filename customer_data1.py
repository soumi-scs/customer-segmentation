# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 18:35:48 2024

@author: dolls
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Read the datasets
customer_data = pd.read_csv("C:/Users/dolls/Desktop/Mall_Customers.csv")
print(customer_data)

# Extracting some important features
x = customer_data.iloc[:, [3, 4]].values
print(x)

# Choosing the correct cluster using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow method graph
sns.set()
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Optimum number of clusters n=5
# Training the K-Means clustering model
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
# Return a label for each data point based on their cluster
y = kmeans.fit_predict(x)
print(y)

# Visualizing all the clusters
plt.figure(figsize=(8, 8))
plt.scatter(x[y == 0, 0], x[y == 0, 1], s=50, c='green', label='Cluster 1')
plt.scatter(x[y == 1, 0], x[y == 1, 1], s=50, c='blue', label='Cluster 2')
plt.scatter(x[y == 2, 0], x[y == 2, 1], s=50, c='yellow', label='Cluster 3')
plt.scatter(x[y == 3, 0], x[y == 3, 1], s=50, c='red', label='Cluster 4')
plt.scatter(x[y == 4, 0], x[y == 4, 1], s=50, c='violet', label='Cluster 5')

# Plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')

# Adding labels to each cluster's centroid
cluster_names = ['Low Income, Low Spend', 'Medium Income, Medium Spend', 'High Income, High Spend', 'Low Income, High Spend', 'High Income, Low Spend']
for i, name in enumerate(cluster_names):
    plt.text(kmeans.cluster_centers_[i, 0], kmeans.cluster_centers_[i, 1], name, fontsize=15, ha='center')

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
