# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 20:47:28 2023

@author: USER
"""

import numpy as np
import pickle
import streamlit as st


#loading the saved model
loaded_model = pickle.load(open("C:\Users\USER\trained_model.sav","rb"))

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
    
    #getting the input data from the user
    
    
    
    fixed_acidity = st.text_input("Quantity of Fixed Acidity")
    volatile_acidity = st.text_input("Quantity of Volatile Acidity")
    citric_acid = st.text_input("Quantity of Citric Acid")
    residual_sugar = st.text_input("Quantity of Residual Sugar")
    chlorides = st.text_input("Quantity of Chlorides")
    free_sulfur_dioxide = st.text_input("Quantity of Free Sulfur Dioxide")
    total_sulfur_dioxide = st.text_input("Quantity of Total Sulfur Dioxide")
    density = st.text_input("Density")
    pH = st.text_input("Value of pH")
    sulphates = st.text_input("Quantity of Suplhates")
    alcohol = st.text_input("Quantity of alcohol")
    
    
    
    #code for prediction
    Quality = ''
    
    #creating a button for prediction
    
    if st.button("Wine Quality Results"):
        Quality = wine_quality_prediction([fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol])
        
    
    st.success(Quality)
    
    
    
    uploaded_file = st.file_uploader(
    "Choose your database", accept_multiple_files=False)
if uploaded_file is not None:
    file_name = uploaded_file
else:
    file_name = open(r"C:\Users\USER\Downloads\New folder\winequalityN.csv")
    
    
    
    
    
    
if __name__ == '__main__':
    main()
