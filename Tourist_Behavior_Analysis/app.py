import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pandas as pd 
import seaborn as sns 
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt 



# Modify app name and Icon
# Config Function 
st.set_page_config(page_title='Tourist Behavior', page_icon='✈', layout='wide', initial_sidebar_state='auto')



# Hide Menu and Footer
hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """   
st.markdown(hide_menu_style, unsafe_allow_html=True)


st.title("Tourist Behavior ✈")
st.markdown('---')

with st.sidebar.header("Upload you dataset (.csv)"):
    uploaded_file = st.sidebar.file_uploader('Upload your file', type=['csv'])
    df = sns.load_dataset('titanic')
    
Machine_Learning_Model_name = st.sidebar.selectbox('Select Model', (
    'Linear R', 'Decision tree R', 'KNN R', 'SVR R'))

    # profiling report for pandas

if uploaded_file is not None:
    #@st.cache
    def load_data():
        csv = pd.read_csv(uploaded_file, encoding='latin-1')
        return csv
    df = load_data()


df = pd.read_csv("tourist_behavior.csv")

with st.sidebar:
    selected = option_menu(
        menu_title='Sahil Ali',
        options= ['Introduction', 'Dataset','Visualization'],
        icons = ['cursor-fill', 'clipboard-data','graph-up-arrow'],
        
        menu_icon = 'apple',
        default_index = 0,
        orientation = 'vertical',
    )


if selected == 'Introduction':
    st.title("🌟Introduction🌟")
    st.markdown('---')
   
    c1 , c2 = st.columns(2)
    img1 = Image.open('images/img1.png')
    img2 = Image.open('images/img2.png')
    with c1:
        st.image(img1, width=500)
    with c2:
        st.image(img2, width=300)
        st.markdown('#### **CONNECT ME**')
        st.markdown('##### [![LinkedIn](https://brand.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg)](https://www.linkedin.com/in/sahil-jatoi-ba10731b3/)')
    st.markdown('''
    #### This Web-App is developed by **Sahil Ali** as a part of his **Big Data Analytics internship at NAVTTC**.
    #### This project is based on analyze the tourist behavior in the year (2010-2016). The data is collected from the website of the United Nations World Tourism Organization (UNWTO). The data is collected from 185 countries and 6 regions.''')
    


if selected == 'Dataset':
    st.title("🌟Dataset🌟")
    st.markdown('---')
    c3 , c4 = st.columns(2)
    with c3:
        st.dataframe(df)
    with c4:
        img3 = Image.open('images/img3.png')
        st.image(img3, width=300)
        
    st.markdown('---')
    rows, columns =df.shape
    st.markdown('### **The total number of the rows are**')
    st.success(rows)
    st.markdown('### **The total number of the columns are**')
    st.success(columns)
    
    st.markdown('---')
    st.markdown('### **Select the columns to see the data detail**')
    option = st.selectbox('', ['Data structure','Missing values', 'Unique values'])
    if option == 'Data structure':
        st.markdown('### **The data structure is**')
        st.dataframe(df.describe().T)
        
    if option == 'Missing values':
        st.markdown('### **The missing values are**')
        st.dataframe(df.isnull().sum())
        
    if option == 'Unique values':
        st.markdown('### **The unique values are**')
        st.dataframe(df.nunique())
        

if selected == 'Visualization':
    st.title("🌟Visualization🌟")
    st.markdown('---') 
    
    st.header('Pie Chart')
    year_wise_sum = df.groupby('year')['tourists'].sum().reset_index().sort_values(by='tourists',ascending=False,ignore_index=True)
    percent = px.pie(year_wise_sum, values='tourists', names='year', 
                 color_discrete_sequence=px.colors.sequential.RdBu, 
                color='year',labels={'tourists':'Tourists'})                 
    percent.update_layout(title_text='Year wise tourists',title_x=0.5,font=dict(size=18),
                        legend=dict(font=dict(size=18)),margin=dict(l=20,r=20,b=20,t=50))
    st.write(percent)
    st.markdown('### **The above pie chart shows that the number of tourist from the country is increasing year by year with percentage vise 2010- 2011- 2012- 2013- 2014- 2015- 2016**')
    st.markdown('---')
    
    st.header('Bar Chart')
    group2 = df.groupby(['year',
                     'month'])['tourists'].sum().reset_index().sort_values(by='year',ascending=False,ignore_index=True)
    
    fig2 = px.bar(group2, x='year', y='tourists', color='month',barmode='group')
    fig2.update_layout(title_text='Year wise tourists',title_x=0.5,font=dict(size=18))
    st.write(fig2)
    st.markdown('### **The above bar chart shows the number of tourist with respect to year and month**')
    st.markdown('---')
        
    st.header('Histogram')
    group3 = df.groupby(['nationality'])['tourists'].mean().reset_index().sort_values(by='tourists',ascending=False,ignore_index=True)
    
    fig3 = px.histogram(group3.head(30), x='nationality', y='tourists', color='nationality')
    fig3.update_layout(title_text='Nationality wise rate (top 30)',title_x=0.5,font=dict(size=18),
                    xaxis_title='National',yaxis_title='Tourists',margin=dict(l=20,r=20,b=20,t=50))
    st.write(fig3)
    st.markdown('### **The above histogram shows the top 30% number of tourist in nation.**')
    
    st.markdown('---')
    st.header('Histogram')
    fig4 = px.histogram(group3.tail(30), x='nationality', y='tourists', color='nationality')
    fig4.update_layout(title_text='Nationality wise rate(Below 30)',title_x=0.5,font=dict(size=18),
                    xaxis_title='National',yaxis_title='Tourists',margin=dict(l=20,r=20,b=20,t=50))
    
    st.write(fig4)
    st.markdown('### **The above histogram shows the below 30% number of tourist in nation.**')
    st.markdown('---')
    
    
    st.header('3D Scatter Plot')
    fig5 = px.scatter_3d(df, x='year', y='month', z='tourists',color='region',size='tourists')
    fig5.update_layout(title_text='Year, Month and Tourists',title_x=0.5)
    st.write(fig5)
    st.markdown('### **The above 3D-scatter plot shows the number of tourist in different regions with respect to year and month.**')
