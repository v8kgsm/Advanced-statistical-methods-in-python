# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:19:39 2020

@author: Vikas Tiwari
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns   #overrides all graphic methodss
sns.set()

raw_data = pd.read_csv('2.01. Admittance.csv')
print(raw_data)
print("-------------------------------------")
data = raw_data.copy()
data['Admitted'] = data['Admitted'].map({'Yes':1,'No':0})
print(data)
'''
print("-------------------variables------------------")
'''
y = data['Admitted']
x1 = data['SAT']
print("-------------------Scatter Plot------------------")
plt.scatter(x1,y,color='C0')
plt.xlabel('SAT',fontsize=20)
plt.ylabel('Admitted',fontsize=20)
plt.show()

#Plot with a regression line
x = sm.add_constant(x1)
reg_lin = sm.OLS(y,x)
results_lin = reg_lin.fit()
plt.scatter(x1,y,color = 'C0')
y_hat = x1*results_lin.params[1]+results_lin.params[0]

plt.plot(x1,y_hat,lw=2.5,color='C8')
plt.xlabel('SAT',fontsize = 20)
plt.ylabel('Admitted',fontsize = 20)
plt.show()
print("-----------------logistic regression curve--------------------")
#plot with a logistic regression curve
reg_log = sm.Logit(y,x)
results_log = reg_log.fit()

def f(x,b0,b1):
    return np.array(np.exp(b0+x*b1) /(1+ np.exp(b0+x*b1)))

f_sorted = np.sort(f(x1,results_log.params[0],results_log.params[1]))
x_sorted = np.sort(np.array(x1))

plt.scatter(x1,y,color='C0')
plt.xlabel('SAT',fontsize = 20)
plt.ylabel('Admitted',fontsize = 20)
plt.plot(x_sorted,f_sorted,color='C8')
plt.show()
#the score between 1600 and 1750 is uncertain
#the function shows the probability of admission, given an SAT score