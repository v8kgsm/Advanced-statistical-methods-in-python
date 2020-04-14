# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:33:30 2020

@author: Vikas Tiwari
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   #overrides all graphic methodss
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
data = pd.read_csv('1.02. Multiple linear regression.csv')
print(data.head())
print("-------------------------------")
print(data.describe())
print("-------------------------------")
#declare dependent and independent variable
x = data[['SAT','Rand 1,2,3']]
y = data['GPA']
scaler = StandardScaler()
print(scaler.fit(x))
print("-------------------------------")
x_scaled = scaler.transform(x)
print(x_scaled)
print("-------------------------------")
#regression with scaled features
reg = LinearRegression()
print(reg.fit(x_scaled,y))
print("-------------------------------")
print("Coefficients:",reg.coef_)
print("Intercept:",reg.intercept_)
print("-------------------------------")
reg_summary = pd.DataFrame([['Intercept/Bias'],['SAT'],['Rand 1,2,3']],columns=['Features'])
reg_summary['weights(Coefficients)'] = reg.intercept_,reg.coef_[0],reg.coef_[1]
print(reg_summary)
#the bigger the weight the bigger the impact
#the ML word for intercept is bias
print("--------------Make predictions with the standardized coefficients!!-----------------")
new_data = pd.DataFrame(data=[[1700,2],[1800,1]],columns=['SAT','Rand 1,2,3'])
print(new_data)
print("Unstandard:",reg.predict(new_data))
print("-------------------------------")
new_data_scaled = scaler.transform(new_data)
print(new_data_scaled)
print("-------------------------------")
print("Prediction 1:",reg.predict(new_data_scaled))
print("--------------Removed Rand 1,2,3-----------------")
reg_simple = LinearRegression()
x_simple_matrix = x_scaled[:,0].reshape(-1,1)
print(reg_simple.fit(x_simple_matrix,y))
print("Prediction 2:",reg_simple.predict(new_data_scaled[:,0].reshape(-1,1)))
print("Rounded values in both comes closer to 3.09 , 3.26(prediction 1 and 2)")







