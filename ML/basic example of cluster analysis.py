# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:35:47 2020

@author: Vikas Tiwari
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns   #overrides all graphic methodss
sns.set()
from sklearn.cluster import KMeans
data = pd.read_csv('3.01. Country clusters.csv')
print(data)
print("-------------------------------------------------")

#map the data
data_mapped = data.copy()
data_mapped['Language'] = data_mapped['Language'].map({'English':0,'French':1,'German':2})
print(data_mapped)
x=data_mapped.iloc[:,1:4]
print(x)
print("-------------------------------------------------")
kmeans = KMeans(3)
kmeans.fit(x)

#clustering
identified_clusters = kmeans.fit_predict(x)
print(identified_clusters)
print("-------------------------------------------------")
data_with_clusters = data_mapped.copy()
data_with_clusters['Cluster'] = identified_clusters
print(data_with_clusters)
print("-------------------------------------------------")
plt.scatter(data_with_clusters['Longitude'],data_with_clusters['Latitude'],c=data_with_clusters['Cluster'],cmap='rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()
#calculating custers and With in cluster sum of squares(wcss)
'''
the minimum the wcss the perfect the clustering solution is
'''
print("-------------------------------------------------")
#wcss
print(kmeans.inertia_)
#loop
print("-------------------------------------------------")
wcss=[]
for i in range(1,7):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss_iter =     kmeans.inertia_
    wcss.append(wcss_iter)
print(wcss)
print("-------------------------------------------------")
#the elbow mathod
number_clusters = range(1,7)
plt.plot(number_clusters,wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('with-in cluster sum of squares')