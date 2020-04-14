# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:03:06 2020

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns   #overrides all graphic methodss
sns.set()

data=pd.read_csv('Assumption_3.csv')        #data is missing
data['log_x'] = np.log(data['x'])
data['log_y'] = np.log(data['y'])
plt.scatter(data['x'],data['y'])
plt.xlabel('X')
plt.ylabel('Y')

#2
#semilog model
plt.scatter(data['log_x'],data['y'])
plt.xlabel('Log X')
plt.ylabel('Y')

#3semilog model
plt.scatter(data['x'],data['log_y'])
plt.xlabel('X')
plt.ylabel('log Y')

#4 log-log model
plt.scatter(data['log_x'],data['log_y'])
plt.xlabel('log X')
plt.ylabel('log Y')

