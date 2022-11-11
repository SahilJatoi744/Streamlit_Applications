# Importing libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,precision_score,f1_score

# load dataset
df = pd.read_csv("diabetes.csv")
# title 
st.title('Diabetes Detection')

#side bar
st.sidebar.header('Patient Data')
st.subheader('Description Stats of Data')
st.write(df.describe())

# Data Split into X and Y
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Function 
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    sk = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0.0, 846.0, 30.5)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('Age', 21, 81, 29)
    
    # Dictionary    
    user_report_data = { "Pregnancies": pregnancies,
                        "Glucose": glucose,
                        "Blood Pressure": bp,
                        "Skin Thickness": sk,
                        "Insulin": insulin,
                        "BMI": bmi,
                        "Diabetes Pedigree Function": dpf,
                        "Age": age
    }

    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data
# Patient Data
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# model
rc = RandomForestClassifier()
rc.fit(X_train, y_train)
user_result = rc.predict(user_data)

# Visualize
st.title('Visualization')

# Color Function
if user_result[0]==0:
    color = 'green'
else:
    color= 'red'

df.columns
list = ['Pregnancies','Glucose','Insulin','BMI','Age']
# Age vs Other variables
for i in list[0:7]:
    st.header(i+" Count Graph (Other vs Yours)")
    fig_preg = plt.figure()
    ax1 = sns.scatterplot(x='Age', y=i, data=df, hue='Outcome', palette='Set1')
    ax2 = sns.scatterplot(x=user_data['Age'], y=user_data[i], color=color)
    # plt.xticks(np.arange(0, 100, 10))
    # plt.yticks(np.arange(0, 20, 2))
    # 
    plt.title('Age vs '+i)
    st.pyplot(fig_preg)

# output
st.header("Your Report :")
output = ''
if user_result[0] ==0:
    output = "You don't have Diabetes"
    st.balloons()
else:
    output = "You have Diabetes"
    st.snow()
    #st.warning("Sugar, Sugar, Sugar")
    #st.error('Error message')
    #st.exception("Exception")
st.title(output)
st.subheader("Accuracy Score :")
st.write(str(accuracy_score(y_test, rc.predict(X_test))*100)+'%')

st.subheader("Confusion Matrix :")
fig_preg2 = plt.figure()
sns.heatmap(confusion_matrix(y_test, rc.predict(X_test)), annot=True, cmap='Blues')
st.pyplot(fig_preg2)
st.subheader("Recall Score :")
st.write(str(recall_score(y_test, rc.predict(X_test))*100)+'%')

st.subheader("Precision Score :")
st.write(str(precision_score(y_test, rc.predict(X_test))))

st.subheader("F1 Score :")
st.write(str(f1_score(y_test, rc.predict(X_test))))
