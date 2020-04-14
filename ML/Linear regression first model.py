# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:30:13 2020

@author: Vikas Tiwari
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns   #overrides all graphic methodss
sns.set()
#equation 
#y=b0*x0+b1*x1

#simple regression model

data = pd.read_csv('1.01. Simple linear regression.csv')
print(data.describe())
y=data['GPA']
x1 = data['SAT']
plt.scatter(x1,y)
plt.xlabel('SAT',fontsize=20)
plt.ylabel('GPA',fontsize=20)
plt.show()
x=sm.add_constant(x1)
results=sm.OLS(y,x).fit()
print(results.summary())

#ploting fitting line
plt.scatter(x1,y)
yhat=0.0017*x1 + 0.275
fig = plt.plot(x1,yhat,lw=4,c='orange',label='regression line')
plt.xlabel('SAT',fontsize=20)
plt.ylabel('GPA',fontsize=20)
plt.show()