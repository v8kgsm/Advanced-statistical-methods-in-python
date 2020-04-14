# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:23:28 2020

@author: Vikas Tiwari
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns   #overrides all graphic methodss
sns.set()
from sklearn.cluster import KMeans
data = pd.read_csv('3.12. Example.csv')
print(data)

#ploting
plt.scatter(data['Satisfaction'],data['Loyalty'])
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
print('-------------------------------------------------------')
x=data.copy()
kmeans = KMeans(2)
print(kmeans.fit(x))
#clustering results
clusters = x.copy()
clusters['cluster_pred'] = kmeans.fit_predict(x)
plt.scatter(clusters['Satisfaction'],clusters['Loyalty'],c=clusters['cluster_pred'],cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
print('---------------------------standardisation----------------------------')
from sklearn import preprocessing   
x_scaled = preprocessing.scale(x)
print(x_scaled)
# =============================================================================
# since we dont know the number od clusters needed we will take help
# of elbow method.
# =============================================================================
wcss=[]
for i in range(1,10):
    kmeans = KMeans(1)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)
print(wcss)
print('-------------------------------------------------------')
plt.plot(range(1,10),wcss)
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
