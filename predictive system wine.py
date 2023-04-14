# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle

#loading the saved model
loaded_model = pickle.load(open(r'C:\Users\USER\trained_model.sav','rb'))


data = (4.2,0.215,0.23,1.808289,0.041,64.0,157.0,0.99688,3.42,0.44,8.0)
array = np.asarray(data)
final = array.reshape(1,-1)
prediction = loaded_model.predict(final)
print(prediction)
if (prediction[0] == 1):
    print('Wine Quality is Good')
else: 
    print('Wine Quality is Bad')
