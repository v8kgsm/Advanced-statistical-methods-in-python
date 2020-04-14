# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:20:12 2020

@author: Vikas Tiwari
"""
'''
As explained in the lesson, the concept 'normalization' has different meanings in different contexts.

There are two materials which we find particularly useful:

1) The Wikipedia article on Feature scaling: https://en.wikipedia.org/wiki/Feature_scaling

2) This article on L1-norm and L2-norm: http://www.chioka.in/differences-between-the-l1-norm-and-the-l2-norm-least-absolute-deviations-and-least-squares/
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   #overrides all graphic methodss
sns.set()
from sklearn.linear_model import LinearRegression
data = pd.read_csv('1.01. Simple linear regression.csv')
print(data.head())
x= data['SAT']
y = data['GPA']
x_matrix = x.values.reshape(-1,1)
reg=LinearRegression()
print("Regression:",reg.fit(x_matrix,y))
print("R-Squared:",reg.score(x_matrix,y))
print("Coefficients:",reg.coef_)
print("Intercept:",reg.intercept_)

print("-------------------------------")
#print("Predict:",reg.predict(1740))
new_data=pd.DataFrame(data=[1740,1760],columns=['SAT'])
print(new_data)
print("-------------------------------")
print(reg.predict(new_data))
new_data['Predicted_GPA']=reg.predict(new_data)
print("-------------------------------")
print(new_data)
print("-------------------------------")

plt.scatter(x,y)
yhat=reg.coef_*x_matrix + reg.intercept_
fig=plt.plot(x,yhat,lw=4,c='orange',label='regression line')
plt.xlabel('SAT',fontsize=20)
plt.ylabel('GPA',fontsize=20)
plt.show()