# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:52:07 2020

@author: Vikas Tiwari
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns   #overrides all graphic methodss
sns.set()

#apply fix to the statsmodel library
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

raw_data = pd.read_csv('2.01. Admittance.csv')
print(raw_data)
print("-------------------------------------")
data = raw_data.copy()
data['Admitted'] = data['Admitted'].map({'Yes':1,'No':0})
print(data)
y = data['Admitted']
x1 = data['SAT']
#regression

x=sm.add_constant(x1)
reg_log = sm.Logit(y,x)
print("-----------------------it has maximum 35 iteration after that it will terminate the loop--------------------------")
results_log = reg_log.fit()
print(data.describe())
print(results_log.summary())
print("-------------------------------------")
x0 = np.ones(168)
reg_log = sm.Logit(y,x0)
results_log = reg_log.fit()
print(results_log.summary())





