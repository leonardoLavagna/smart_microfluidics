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

file_path = 'data.csv'
data = pd.read_csv(file_path, encoding='latin1')
data = data.set_index('ID')
st.write("Dataset Preview:", data.head())

def get_user_input(features):
    input_values = {}
    st.write("Enter values for the following features:")
    
    # Loop through features to get input from the user
    for feature in features:
        # Number input for each feature
        input_values[feature] = st.number_input(f"{feature}", value=0.0)
        
    return input_values

def predict_size_pdi(input_values, model, feature_columns):
    input_array = np.array([input_values[feature] for feature in feature_columns]).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]
    
def main():
    # Load pre-trained model (assuming the model is saved as 'trained_model.pkl')
    try:
        with open('trained_model.pkl', 'rb') as file:
            model = pickle.load(file)
        st.write("Model loaded successfully!")
    except FileNotFoundError:
        st.error("Model file not found. Please train and save the model first.")
        return

    # Define feature columns (adjust this based on your trained model's features)
    feature_columns = ['feature1', 'feature2', 'feature3', 'feature4']  # Example feature names
    st.write(f"Features: {feature_columns}")
    
    # User input section
    st.header("Predict 'Size' and 'PDI'")
    user_input = get_user_input(feature_columns)
    
    # Button to make prediction
    if st.button("Predict"):
        prediction = predict_size_pdi(user_input, model, feature_columns)
        st.write(f"Predicted 'Size': {prediction[0]:.2f}")
        st.write(f"Predicted 'PDI': {prediction[1]:.2f}")

if __name__ == "__main__":
    main()

