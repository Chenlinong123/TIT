#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np 
import pandas as pd 
import shap
import joblib
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


# In[ ]:


background_data = pd.read_csv('background_data.csv')


# In[ ]:


plt.style.use('default')


# In[ ]:


st.markdown("<h1 style='text-align: center; color: black;'>Online Interpretable Machine Learning Models For Dynamic Prediction Of Thrombocytopenia Risk In Acute Ischemic Stroke Patients Undergoing Tirofiban-assisted Mechanical Thrombectomy</h1>", unsafe_allow_html=True)


# In[ ]:


def user_input_features():
    st.sidebar.header('Make a prediction')
    st.sidebar.write('User input parameters below ⬇️')
    a1 = st.sidebar.number_input("age")
    a2 = st.sidebar.number_input("diastolic blood pressure")
    a3 = st.sidebar.number_input("platelets (*10^9)")
    a4 = st.sidebar.number_input("INR")
    a5 = st.sidebar.number_input("serum creatinine")
    a6 = st.sidebar.number_input("triglyceride")
    a7 = st.sidebar.number_input("thrombectomy passes")
    output = [a1,a2,a3,a4,a5,a6,a7]
    return output

outputdf = user_input_features()


# In[ ]:


# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    ANN = load_model('ANN_model.h5')
    #标准化
    a1 = outputdf[0]
    a2 = outputdf[1]
    a3 = outputdf[2]
    a4 = outputdf[3]
    a5 = outputdf[4]
    a6 = outputdf[5]
    a7 = outputdf[6]
    
    
    # Store inputs into dataframe
    a1s=(a1-66.87)/11.982
    a2s=(a2-82.21)/14.389
    a3s=(a3-204.82)/44.218
    a4s=(a4-1.0006)/0.0769
    a5s=(a5-71.9399)/20.98994
    a6s=(a6-1.2196)/0.55119
    a7s=(a7-2.15)/1.203
    
    stdf = [a1s,a2s,a3s,a4s,a5s,a6s,a7s]
    
    X = pd.DataFrame([outputdf], columns= ["age","DBP","platelet","INR","Scr","TG","passes"])
    X_standard = pd.DataFrame([stdf], columns= ["age","DBP","platelet","INR","Scr","TG","passes"])
    # Get prediction
    p1 = ANN.predict(X_standard)[0]
    p2 = ANN.predict(X_standard)[0,-1]
    m1 = round(float(p2) * 100, 2)
    p3 = "%.2f%%" % (m1)

    # Output prediction
    st.write(f'Predicted results: {p1}')
    st.write('0️⃣ means non-TIT, 1️⃣ means TIT')
    st.text(f"Prediction probabilities：1️⃣ {p3}")
    
    #SHAP
    st.title('SHAP')
    
    #个例
    
    
    explainer_ANN = shap.KernelExplainer(ANN.predict,background_data)
    shap_values= explainer_ANN.shap_values(X_standard)
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.force_plot(explainer_ANN.expected_value,shap_values[0],X.iloc[0],matplotlib=True)

    #shap瀑布图
    #shap_values2 = explainer_xgb(X_standard) 
    #shap.plots.waterfall(shap_values2[0])
    st.pyplot(bbox_inches='tight')
    


# In[ ]:



