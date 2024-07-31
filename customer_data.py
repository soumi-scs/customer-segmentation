# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 11:54:12 2024

@author: dolls
"""
#importing libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#read the datasets

customer_data=pd.read_csv("C:/Users/dolls/Desktop/Mall_Customers.csv")
print(customer_data)

''' load the first 5 rows of the datasets
customer_data=customer_data.head(5)
print(customer_data)

#finding the rows and columns
customer_data=customer_data.shape
print(customer_data)

 

# Getting the count of null values in each column
null_values = customer_data.isnull().sum()

# Printing the result
print(null_values)'''

#extracting some imp features
x=customer_data.iloc[:,[3,4]].values
print(x)
#choosing the correct cluster
#WCSS->within cluster sum of squrares
#finding wcss values using elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# Plotting the elbow method graph
sns.set()
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#optimum number of clusters n=5;
#Trainning the k-means clustering model
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
#return a label for each data points based on their clusters
y=kmeans.fit_predict(x)
print(y)
#visualizing all the clusters

plt.figure(figsize=(8,8))
plt.scatter(x[y==0,0],x[y==0,1],s=50,c='green',label='cluster1')
plt.scatter(x[y==1,0],x[y==1,1],s=50,c='blue',label='cluster2')
plt.scatter(x[y==2,0],x[y==2,1],s=50,c='yellow',label='cluster3')
plt.scatter(x[y==3,0],x[y==3,1],s=50,c='red',label='cluster4')
plt.scatter(x[y==4,0],x[y==4,1],s=50,c='violet',label='cluster5')

#plot the centroid
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='cyan',label='centroid')
plt.title('customer groups')
plt.xlabel('Annual income')
plt.ylabel('spending score')
plt.show


    