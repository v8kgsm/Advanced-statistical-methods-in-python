# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:51:26 2020

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

raw_data = pd.read_csv('2.02. Binary predictors.csv')
print(raw_data)

data = raw_data.copy()
data['Admitted'] = data['Admitted'].map({'Yes':1,'No':0})
data['Gender'] = data['Gender'].map({'Female':1,'Male':0})

y = data['Admitted']
x1 = data[['SAT','Gender']]

x=sm.add_constant(x1)
reg_log = sm.Logit(y,x)

results_log = reg_log.fit()
#print(data.describe())
print(results_log.summary())
#probabilities 
print("--------------------------Predicted values by the models--------------------------------")
np.set_printoptions(formatter={'float':lambda x:"{0:0.2f}".format(x)})
print(results_log.predict())
print("------------Actual Values------------")
print(np.array(data['Admitted']))
print("----------------------------Confusion Matrix--------------------------")
print(results_log.pred_table())
print("Formatted in a specified manner")
cm_df = pd.DataFrame(results_log.pred_table())
cm_df.columns = ['Predicted 0' , 'predicted 1' ]
cm_df = cm_df.rename(index={0:'Actual 0',1:'Actual 1'})
print(cm_df)

print("------------Accuracy of the model------------")

cm = np.array(cm_df)
accuracy_train = (cm[0,0]+cm[1,1])/cm.sum()
print(accuracy_train)

print("----------testing the data--------------")
test = pd.read_csv('2.03. Test dataset.csv')
test['Admitted'] = test['Admitted'].map({'Yes':1,'No':0})
test['Gender'] = test['Gender'].map({'Female':1,'Male':0})
test_actual = test['Admitted']
test_data = test.drop(['Admitted'],axis=1)
test_data = sm.add_constant(test_data)
print(test_data)

def confusion_matrix(data,actual_values,model):
    pred_values = model.predict(data)
    bins = np.array([0,0.5,1])
    cm = np.histogram2d(actual_values,pred_values,bins=bins)[0]
    accuracy = (cm[0,0]+cm[1,1])/cm.sum()
    return cm,accuracy
cm = confusion_matrix(test_data,test_actual,results_log)
print(cm)
cm_df = pd.DataFrame(cm[0])
cm_df.columns = ['Predicted 0' , 'predicted 1' ]
cm_df = cm_df.rename(index={0:'Actual 0',1:'Actual 1'})
print(cm_df)
print("Missclassification Rate:",str((1+1)/19))





