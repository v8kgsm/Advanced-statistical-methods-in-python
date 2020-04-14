# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 09:24:17 2020

@author: Vikas Tiwari
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns   #overrides all graphic methodss
sns.set()
#equation
#GPA = b0 + b1SAT + b2Rand1,2,3
data = pd.read_csv('1.02. Multiple linear regression.csv')
print(data)
y=data['GPA']
x1 = data[['SAT','Rand 1,2,3']]
x=sm.add_constant(x1)
results=sm.OLS(y,x).fit()
print(results.summary())