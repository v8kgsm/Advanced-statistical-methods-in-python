# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:35:50 2020

@author: Vikas Tiwari
"""
import numpy as np
from sklearn.model_selection import train_test_split
a = np.arange(1,101)
print(a)
b = np.arange(501,600)
print(train_test_split(a))
#a_train,a_test = train_test_split(a , test_size=0.2 , shuffle=False)
a_train,a_test = train_test_split(a , test_size=0.2 , random_state= 365)

#to randomise data with same randomness we use random_state
# shuffle is default true we can set it to False.
 #to split the data into training and testing (test_size) 
print(a_train.shape,a_test.shape)
print("Training Data:",a_train)
print("Testing Data:",a_test)
#we can do same for b array where after shuffling we will find some similarity in the randomness


