# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 20:47:28 2023

@author: USER
"""

import numpy as np
import pickle
import streamlit as st


#loading the saved model
loaded_model = pickle.load(open('trained_model.sav','rb'))

#creating a function for prediction
def wine_quality_prediction(data):
    
    array = np.asarray(data)
    final = array.reshape(1,-1)
    prediction = loaded_model.predict(final)
    print(prediction)
    if (prediction[0] == 1):
        return 'Wine Quality is Good'
    else: 
        return 'Wine Quality is Bad'
    
    
    
    
def main():
    
    #giving a title
    st.title('Wine Quality Prediction Web App')
    st.caption('By SHRUTI KHANDELWAL')
    
    #getting the input data from the user
    
    
    
    fixed_acidity = st.slider("Quantity of Fixed Acidity",1.0,20.0)
    volatile_acidity = st.slider("Quantity of Volatile Acidity",0.0,2.0)
    citric_acid = st.slider("Quantity of Citric Acid",0.0,5.0)
    residual_sugar = st.slider("Quantity of Residual Sugar",0.0,20.0)
    chlorides = st.slider("Quantity of Chlorides",0.0,2.0)
    free_sulfur_dioxide = st.slider("Quantity of Free Sulfur Dioxide",0.0,100.0)
    total_sulfur_dioxide = st.slider("Quantity of Total Sulfur Dioxide",0.0,200.0)
    density = st.slider("Density",0.0,1.0)
    pH = st.slider("Value of pH",0.0,14.0)
    sulphates = st.slider("Quantity of Sulphates",0.0,10.0)
    alcohol = st.slider("Quantity of alcohol",0.0,20.0)
    
    
    
    #code for prediction
    Quality = ''
    
    #creating a button for prediction
    
    if st.button("Wine Quality Results"):
        Quality = wine_quality_prediction([fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol])
        
    
    st.success(Quality)
     
        
        

    
    
    
if __name__ == '__main__':
    main()
