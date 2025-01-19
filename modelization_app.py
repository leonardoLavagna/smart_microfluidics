import streamlit as st
import pandas as pd

st.title("Data Modelization Dashboard")

file_path = 'data.csv'
data = pd.read_csv(file_path, encoding='latin1')
data = data.set_index('ID')
st.write("Dataset Preview:", data.head())
