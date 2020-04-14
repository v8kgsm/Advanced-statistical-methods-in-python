# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:00:47 2020

@author: Vikas Tiwari
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   #overrides all graphic methodss
sns.set()
from sklearn.linear_model import LinearRegression

raw_data = pd.read_csv('1.04. Real-life example.csv')
print(raw_data.head())

#view each result individually
#descriptive stats
print(raw_data . describe(include='all'))
#axis 0 is for rows
#axis 1 is for columns
data = raw_data.drop(['Model'],axis=1)
print(data . describe(include='all'))

#dealing with missing values
print(data.isnull().sum())

#removing missing values
data_no_mv = data.dropna(axis=0)
print(data_no_mv.describe(include='all'))

#exploring with PDF'S
print(sns.distplot(data_no_mv['Price']))

#dealing with outliers
q = data_no_mv['Price'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['Price']<q]
print(data_1.describe(include='all'))
print(sns.distplot(data_no_mv['Price']))    #with lesser outliers
print(sns.distplot(data_1['Price']))
#print(sns.distplot(data_no_mv['Mileage'])) #####
#we can view graph for all entities.

#run Independently each.
#data clearing
q=data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']<q]
print(sns.distplot(data_2['Mileage'])) #after
print(sns.distplot(data_no_mv['Mileage']))  #before

#2
data_3 = data_2[data_2['EngineV']<6.5]
print(sns.distplot(data_3['EngineV']))

q=data_3['Year'].quantile(0.01)
data_4 = data_3[data_3['Year']>q]
print(sns.distplot(data_4['Year'])) #after
print(sns.distplot(data_no_mv['Year']))

data_cleaned = data_4.reset_index(drop=True)
print(data_cleaned.describe(include='all'))

f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True,figsize = (15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['Price'])
print(ax1.set_title('Price and Year'))
ax2.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
print(ax1.set_title('Price and EngineV'))
ax3.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
print(ax1.set_title('Price and Mileage'))
    
print(sns.distplot(data_cleaned['Price']))  #plot of the price

#relaxing the assumptions
log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price'] = log_price
print(data_cleaned)

#ploting log_price
f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True,figsize = (15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['log_price'])
print(ax1.set_title('log Price and Year'))
ax2.scatter(data_cleaned['EngineV'],data_cleaned['log_price'])
print(ax1.set_title('log Price and EngineV'))
ax3.scatter(data_cleaned['Mileage'],data_cleaned['log_price'])
print(ax1.set_title('log Price and Mileage'))
print("------------------------------------------")
data_cleaned = data_cleaned.drop(['Price'],axis=1)
print(data_cleaned.columns.values)

#one of the best ways to check for multicollinearity is through VIF(variance inflation Factor)
'''
VIF=1 : no multicollinearity
1 < VIF < 5 :perfectly okay
5,6,10 < VIF :Unacceptable
5,6,10 varies according to the data and use
'''

from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['Mileage','Year','EngineV']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range (variables.shape[1])]
vif["features"] = variables.columns
print(vif)

data_no_multicollinearity = data_cleaned.drop(['Year'],axis=1)
#dealing with dummy variables
data_with_dummies = pd.get_dummies(data_no_multicollinearity , drop_first=True)
print(data_with_dummies.head())
print(data_with_dummies.columns.values)
cols=['log_price', 'Mileage', 'EngineV', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']
data_preprocessed = data_with_dummies[cols]
print(data_preprocessed.head())

#Linear Regression Model
## Declare the inputs and targets
targets = data_preprocessed['log_price']
inputs = data_preprocessed.drop(['log_price'],axis=1)

#scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print("----------------------------------------")
print(scaler.fit(inputs))
inputs_scaled = scaler.transform(inputs)
#it is not usually recommended to standardise dummy variables
'''
scaling has no effect on the predictive power of dummies,
once scaled though, they will lose all their dummy meaning.
'''
print("----------------------------------------")
#train Test split

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(inputs_scaled ,targets, test_size=0.2 , random_state= 365)

#create Regression
reg=LinearRegression()
print(reg.fit(x_train,y_train))

y_hat = reg.predict(x_train)

plt.scatter(y_train,y_hat)
plt.xlabel("Targets (  y_train )",size=18)
plt.ylabel("Predicitons (  y_hat )",size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()

#residual = differencce between the targets and the predicitions

print(sns.distplot(y_train - y_hat))
plt.title(" Residuals PDF",size=18)

#the residuals are the estimates of the error
print("--------------------R-Squared------------------------------")
print(reg.score(x_train,y_train))
print("----------------------Intercept---------------------------")
print(reg.intercept_)
print("-----------------------Coefficient---------------------------")
print(reg.coef_)
print("--------------------------------------------------")

reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
print(reg_summary)
print("-----------------------Unique Brands---------------------------")
print(data_cleaned['Brand'].unique())

# TESTING
y_hat_test = reg.predict(x_test)
#alpha takes the value from 0 to 1 where 1 being the default
plt.scatter(y_test ,y_hat_test, alpha=0.2)
plt.xlabel("Targets (  y_test )",size=18)
plt.ylabel("Predicitons (  y_hat_test )",size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()
print("--------------------------------------------------")
'''
np.exp(x) returns the exponential of x  
the exponential of log will give the original value of price
'''
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
print(df_pf.head())
#to minimise the randomisation we will overwrite the index
y_test = y_test.reset_index(drop=True)
print(y_test.head())
print("--------------------------------------------------")
df_pf['Target'] = np.exp(y_test)
print(df_pf)
print("--------------------------------------------------")
#optimising the residual is the heart of the  algorithm
df_pf['Residual']=df_pf['Target'] - df_pf['Prediction']
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
print(df_pf)
print("--------------------------------------------------")
print(df_pf.describe())
print("----------------------sorted values----------------------------")
#sorting
#to read whole dataset
pd.options.display.max_rows = 999
pd.set_option('display.float_format',lambda x: ' %.2f' % x) #****
print(df_pf.sort_values(by=['Difference%']))

'''
how to improve our model
1) use a different set of variables
2) remove a bigger part of the outliers
3) use different  kinds of transformation

This is the moment where you really learn.

Take the model we created as a basis and try to improve it.

One of the biggest changes will be observed when you include the 'Model' feature we dropped in the beginning.

Here are some other suggestions:

-> Perform feature selection

-> Create a regression where 'Price' is not transformed

-> Deal with the outliers in a different way
'''

