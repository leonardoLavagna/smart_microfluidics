import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Data Modelization Dashboard")

# Read the dataset
file_path = 'data.csv'
data = pd.read_csv(file_path, encoding='latin1')
data = data.set_index('ID')

# Display dataset preview
st.write("Dataset Preview:", data.head())

# Feature engineering
numerical_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = data.select_dtypes(include=['object']).columns.tolist()
target_columns = ['SIZE', 'PDI']
X = data[numerical_features + categorical_features]

# Function to get user input for features
def get_user_input(features):
    """
    Function to get user input for the required features using Streamlit widgets.
    """
    input_values = {}
    st.write("Enter values for the following features:")
    for feature in features:
        input_values[feature] = st.number_input(f"{feature}", value=0.0)
    return input_values

# Function to predict 'Size' and 'PDI' based on user input
def predict_size_pdi(input_values, model, feature_columns):
    """
    Function to predict 'Size' and 'PDI' based on user input.
    """
    input_array = np.array([input_values[feature] for feature in feature_columns]).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

# Streamlit App
def main():
    # Load the trained model
    try:
        with open('RFR_trained_model.pkl', 'rb') as file:
            model = pickle.load(file)
        st.write("Model loaded successfully!")
    except FileNotFoundError:
        st.error("Model file not found. Please train and save the model first.")
        return

    # Define feature columns
    feature_columns = numerical_features + categorical_features
    st.write(f"Feature columns: {feature_columns}")

    # User input section
    st.header("Predict 'Size' and 'PDI'")
    user_input = get_user_input(feature_columns)

    # Button to trigger prediction
    if st.button("Predict"):
        prediction = predict_size_pdi(user_input, model, feature_columns)
        st.write(f"Predicted 'Size': {prediction[0]:.2f}")
        st.write(f"Predicted 'PDI': {prediction[1]:.2f}")

if __name__ == "__main__":
    main()
