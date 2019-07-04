#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:10:33 2019

@author: xburner
"""

import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation
from sklearn import preprocessing
from math import ceil


#数据获取
data_all = np.load('drive/Smart_City_data/data_all.npy')
print("data_all:")
print(data_all.shape)

data_test = data_all[0:ceil(data_all.shape[0]*0.15),:]
data_train = data_all[ceil(data_all.shape[0]*0.15):ceil(data_all.shape[0]),:]

print("data_test:")
print(data_test.shape)
print("data_train:")
print(data_train.shape)



#数据标签分离
x_train = data_train[:,:-3]
y_train = data_train[:,-3:]



x_test = data_test[:,:-3]
y_test = data_test[:,-3:]


#数据规范化
x_train = preprocessing.scale(x_train)
scaler_x = preprocessing.StandardScaler().fit(x_train) 
x_train = scaler_x.transform(x_train)

y_train = preprocessing.scale(y_train)
scaler_y = preprocessing.StandardScaler().fit(y_train)
y_train = scaler_y.transform(y_train)

x_test = scaler_x.transform(x_test)
y_test = scaler_y.transform(y_test)





model = Sequential()
model.add(Dense(1024, init='uniform', input_dim = 5216,activation = 'relu'))
model.add(Dense(1024, activation = 'relu'))
#model.add(Activation(LeakyReLU(alpha=0.01)))

 
model.add(Dense(512,activation = 'relu'))
model.add(Dense(512,activation = 'relu'))
#model.add(Activation(LeakyReLU(alpha=0.1))) 

model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))

model.add(Dense(32,activation = 'relu'))
model.add(Dense(32,activation = 'relu'))

model.add(Dropout(0.8))

model.add(Dense(3,activation = 'tanh'))
#model.add(Activation(LeakyReLU(alpha=0.01)))


model.summary()

model.compile(loss='mean_squared_error', optimizer="rmsprop", metrics=["mse"])

hist = model.fit(x_train, y_train, batch_size=10, epochs=500,verbose=1,validation_data=[x_test,y_test])

