#IMPORT ALL REQUIRED LIBRARIES

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

df =pd.read_csv('C:/Users/BMS/Desktop/Project_Coding/diabetes.csv')


#HEADINGS
st.title("$*Diabetes Prediction System*$")
st.header("Diabetes Checkup")
st.sidebar.title("$Patient Data$")


st.markdown(
            f'''
            <style>
                .appview-container .css-1544g2n {{
                    padding-top:1rem;
                    background: linear-gradient(to right,  #cbf0ff,#3ca1ff);
                }}
                .appview-container .css-1y4p8pa{{
                    padding-top:2rem;
                   
                }}
                .appview-container {{
                     background: linear-gradient(to right,#b2ffda,#2f80ed );
                }}
                header.css-18ni7ap{{
                    background: linear-gradient(to right, #0909ff,#99f3ff);
                }}
                .css-16idsys p{{
                    color:rgb(171 9 9);
                    font-size:16px;
                }}
                .css-629wbf {{
                background:linear-gradient(to right,#3ca1ff,  #cbf0ff);
                border-radius:0.7rem;
                color:darkblue;
                display: block;
                margin:auto;
                
                }}
                .css-1vbkxwb p{{
                font-size:17px;
                }}
                .css-1629p8f h1, .css-1629p8f h2, .css-1629p8f h3, .css-1629p8f h4, .css-1629p8f h5, .css-1629p8f h6, .css-1629p8f span
                 {{
                 color: rgb(171 9 9);
                 }}
                

                
            </style>
            ''',
            unsafe_allow_html=True,
        )


#X AND Y DATA
x=df.drop('Outcome',axis=1)
y=df.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=0)


#FUCTION
def user_report():
    age=st.sidebar.number_input('Age',value=21,min_value=10,max_value=100)
    pregnancies=st.sidebar.number_input('Pregnancies(0-17)',value=0,min_value=0,max_value=17)
    glucose=st.sidebar.number_input('Glucose( 50-200 mg/dL)',value=71,min_value=50,max_value=200)
    bp=st.sidebar.number_input('Blood Pressure( 40-122 mm Hg)',value=78,min_value=40,max_value=122)
    skinthickness=st.sidebar.number_input('Skin Thickness( 7-100 mm)',value=50,min_value=7,max_value=100)
    insulin=st.sidebar.number_input('Insulin( 0-846 mu U/ml)',value=45,min_value=0,max_value=846)
    bmi=st.sidebar.number_input('BMI( 18.2-67.1 kg/m^2)',value=33.2,min_value=18.2,max_value=67.1,step=0.1)
    dpf=st.sidebar.number_input('Diabetes Pedigree Function(0.08-2.42)',value=0.42,min_value=0.08,max_value=2.42,step=0.01)
    
    user_report_data={
        'Pregnancies':pregnancies,
        'Glucose':glucose,
        'BP':bp,
        'SkinThickness':skinthickness,
        'Insulin':insulin,
        'BMI':bmi,
        'DPF':dpf,
        'Age':age
    }


    report_data=pd.DataFrame(user_report_data, index=[0])
    return report_data

#PATIENT DATA
user_data=user_report()
st.subheader('Patient Data:')
st.write(user_data)


#MODEL BUIDING
x_train=x_train.values
x_test=x_test.values

rf= RandomForestClassifier()
rf.fit(x_train, y_train)
user_result=rf.predict(user_data)


#OUTPUT
st.subheader('Your Report:')

if st.sidebar.button('Predict'):

    if user_result[0]==0:
        output=('<p style=" color:#fbff01; font-size: 50px;">You are not Diabetic,Not to worry.</p>')
        st.markdown(output, unsafe_allow_html=True)
    else:
        output=('<p style=" color:#ff4700; font-size: 50px;">You are Diabetic, You need to consult your doctor.</p>')
        st.markdown(output, unsafe_allow_html=True)
       
    
#VISUALISATIONS

#COLOR FUNCTION
if user_result[0]==0:
    color='blue'
else:
    color='red'


#Age vs Pregnancies
st. header('1.Pregnancy Count Graph(Others vs Yours):')
fig_preg=plt.figure()
ax1=sns.scatterplot(x='Age',y='Pregnancies',data=df,hue='Outcome',palette='Greens')
ax2=sns.scatterplot(x=user_data['Age'],y=user_data['Pregnancies'],s=150,color=color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title('0 - Healthy & 1- Unhealthy')
st.pyplot(fig_preg)

#Age vs Glucose
st. header('2.Glucose Value Graph(Others vs Yours):')
fig_glucose=plt.figure()
ax3=sns.scatterplot(x='Age',y='Glucose',data=df,hue='Outcome',palette='magma')
ax4=sns.scatterplot(x=user_data['Age'],y=user_data['Glucose'],s=150,color=color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,220,10))
plt.title('0 - Healthy & 1- Unhealthy')
st.pyplot(fig_glucose)


#Age vs Bp
st. header('3.Blood Pressure Value Graph(Others vs Yours):')
fig_bp=plt.figure()
ax5=sns.scatterplot(x='Age',y='BloodPressure',data=df,hue='Outcome',palette='Reds')
ax6=sns.scatterplot(x=user_data['Age'],y=user_data['BP'],s=150,color=color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,130,10))
plt.title('0 - Healthy & 1- Unhealthy')
st.pyplot(fig_bp)

#Age vs Skin Thickness
st. header('4.Skin Thickness Value Graph(Others vs Yours):')
fig_st=plt.figure()
ax7=sns.scatterplot(x='Age',y='SkinThickness',data=df,hue='Outcome',palette='Blues')
ax8=sns.scatterplot(x=user_data['Age'],y=user_data['SkinThickness'],s=150,color=color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,110,10))
plt.title('0 - Healthy & 1- Unhealthy')
st.pyplot(fig_st)

#Age vs Insulin
st. header('5.Insulin Value Graph(Others vs Yours):')
fig_i=plt.figure()
ax9=sns.scatterplot(x='Age',y='Insulin',data=df,hue='Outcome',palette='rocket')
ax10=sns.scatterplot(x=user_data['Age'],y=user_data['Insulin'],s=150,color=color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,900,50))
plt.title('0 - Healthy & 1- Unhealthy')
st.pyplot(fig_i)

#Age vs BMI
st. header('6.BMI Value Graph(Others vs Yours):')
fig_bmi=plt.figure()
ax1=sns.scatterplot(x='Age',y='BMI',data=df,hue='Outcome',palette='rainbow')
ax2=sns.scatterplot(x=user_data['Age'],y=user_data['BMI'],s=150,color=color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,70,5))
plt.title('0 - Healthy & 1- Unhealthy')
st.pyplot(fig_bmi)

#Age vs DPF
st. header('7.DPF Value Graph(Others vs Yours):')
fig_dpf=plt.figure()
ax1=sns.scatterplot(x='Age',y='DiabetesPedigreeFunction',data=df,hue='Outcome',palette='YlOrBr')
ax2=sns.scatterplot(x=user_data['Age'],y=user_data['DPF'],s=150,color=color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,3,0.2))
plt.title('0 - Healthy & 1- Unhealthy')
st.pyplot(fig_dpf)





