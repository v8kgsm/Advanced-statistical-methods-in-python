# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 20:08:54 2020

@author: Vikas Tiwari
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   #overrides all graphic methodss
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
data = pd.read_csv('1.02. Multiple linear regression.csv')
print(data.head())
print("-------------------------------")
print(data.describe())
print("-------------------------------")
#declare dependent and independent variable
x = data[['SAT','Rand 1,2,3']]
y = data['GPA']
reg = LinearRegression()
print(reg.fit(x,y))
print("-------------------------------")
print("Coefficients:",reg.coef_)
print("Intercept:",reg.intercept_)
print("-------------------------------")
#calculation R-Squared
print("R-Squared:",reg.score(x,y))
print("-------------------------------")
#formula for adjusted R-Squared 
#R^2 adj = 1-(1-R^2)*(n-1)/(n-p-1)
print(x.shape)
r2=reg.score(x,y)
n=x.shape[0]
p=x.shape[1]
adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
print("adjusted R-Squared:",adjusted_r2)
print("-------------------------------")
#Feature selection through p-values
print(f_regression(x,y))
print("-------------------------------")
p_values = f_regression(x,y)[1]
print(p_values)
print("---------------round off----------------")
print(p_values.round(3))


#creating summary table with p values
print("-----------------creating summary table with p value--------------")
reg_summary = pd.DataFrame(data = x.columns.values,columns=['Features'])
print(reg_summary)
print("-------------------------------")
reg_summary['Coefficients']=reg.coef_
reg_summary['p-values']= p_values.round(3)
print(reg_summary)






