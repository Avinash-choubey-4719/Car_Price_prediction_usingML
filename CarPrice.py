# -*- coding: utf-8 -*-
"""
Created on Sun May  8 11:11:57 2022

@author: DELL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
 

car_dataset = pd.read_csv("car data.csv")


car_dataset.replace({'Fuel_Type':{'Petrol':0, 'Diesel':1, 'CNG':2}}, inplace = True)
car_dataset.replace({'Seller_Type':{'Dealer':0, 'Individual':1}}, inplace = True)
car_dataset.replace({'Transmission':{'Manual':0, 'Automatic':1}}, inplace = True)


x = car_dataset.drop(columns = ['Car_Name', 'Selling_Price'], axis = 1)
y = car_dataset['Selling_Price']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 2)

model = LinearRegression()
model.fit(x_train, y_train)

x_train_prediction = model.predict(x_train)



error_score = metrics.r2_score(y_train, x_train_prediction)


x_test_prediction = model.predict(x_test)

error_test = metrics.r2_score(y_test, x_test_prediction)


plt.scatter(y_train, x_train_prediction)
plt.xlabel("Actual Price")
plt.ylabel("predicted Price")
plt.title("Actual v/s Predicted in training data")
plt.show()



plt.scatter(y_test, x_test_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual v/s predicted in test data")
plt.show()







