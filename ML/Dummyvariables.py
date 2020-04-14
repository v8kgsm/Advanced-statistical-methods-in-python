# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:38:34 2020

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns   #overrides all graphic methodss
sns.set()

raw_data=pd.read_csv('1.03. Dummies.csv')
data = raw_data.copy()
data['Attendance']=data['Attendance'].map({'Yes':1,'No':0})
print(data)
print(data.describe())
y=data['GPA']
x1=data[['SAT','Attendance']]
x=sm.add_constant(x1)
results=sm.OLS(y ,x).fit()
print(results.summary())
plt.scatter(data['SAT'],y,c=data['Attendance'],cmap='RdYlGn_r')
yhat_no = 0.6439 + 0.0014*data['SAT']
yhat_yes = 0.8665 + 0.0014*data['SAT']
yhat=0.0017*data['SAT']+0.275
fig = plt.plot(data['SAT'],yhat_no,lw=2,c='#006837',label='regression line1')
fig = plt.plot(data['SAT'],yhat_yes,lw=2,c='#a50026',label='regression line2')
fig = plt.plot(data['SAT'],yhat,lw=2,c='#4C7280',label='regression line')
plt.xlabel('SAT',fontsize=20)
plt.ylabel('GPA',fontsize=20)
plt.show()               
               
               
                    