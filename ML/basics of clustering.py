# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 20:42:41 2020

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
plt.scatter(data['Longitude'],data['Latitude'])
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()

#select the features
x=data.iloc[:,1:3]

#clustering
#number of clusters required are defined
kmeans = KMeans(3)
kmeans.fit(x)

#clustering results
identified_clusters = kmeans.fit_predict(x)
print("Identified clusters:",identified_clusters)

data_with_clusters = data.copy()
data_with_clusters['Cluster'] = identified_clusters
print(data_with_clusters)
plt.scatter(data_with_clusters['Longitude'],data_with_clusters['Latitude'],c=data_with_clusters['Cluster'],cmap='rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()
