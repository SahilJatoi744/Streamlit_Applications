import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

st.write("HELLO WORLD")
df = px.data.gapminder()
st.write(df.head())