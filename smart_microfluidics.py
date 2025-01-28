import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch
import pickle
from sklearn.ensemble import RandomForestRegressor

# Title of the app
st.title("Smart Microfluidics: Modeling and Visualization")

# Sidebar for navigation
st.sidebar.title("Choose an Option")
section = st.sidebar.selectbox(
    "Section:",
    [
        "Upload Dataset",
        "Modeling",
        "Visualization",
    ],
)

# Upload dataset section
if section == "Upload Dataset":
    st.header("Upload Your Dataset")
    uploaded_file = st.file_uploader("Drag and Drop your CSV file here", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            st.write("Preview of the uploaded data:")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error reading the file: {e}")
    else:
        st.info("Please upload a CSV file.")

# Static file path (for demonstration purposes)
file_path = 'data/data.csv'
try:
    data = pd.read_csv(file_path, encoding='latin1')
    data = data.set_index('ID')
except FileNotFoundError:
    st.warning("Default dataset not found. Please upload your dataset in the 'Upload Dataset' section.")

# Modeling section
if section == "Modeling":
    st.sidebar.title("Model Selection")
    option = st.sidebar.selectbox(
        "Choose a model:",
        [
            "Random forest regressor",
            "XGBoost",
            "Inverse problem",
        ],
    )

    # Random Forest Regressor
    if option == "Random forest regressor":
        st.header("Random Forest Regressor")
        st.write("Using multiple decision trees in parallel for robust predictions of `SIZE` and `PDI`.")
        MODEL_PATH = "models/random_forest_model.pkl"
        try:
            with open(MODEL_PATH, "rb") as file:
                model = pickle.load(file)
            st.write(f"Model loaded successfully from {MODEL_PATH}")

            # Input fields for prediction
            ml = st.selectbox("ML", ["HSPC", "ESM"])
            chip = st.selectbox("CHIP", ["Micromixer", "Droplet junction"])
            esm = st.number_input("ESM", value=0.0, min_value=0.0, max_value=100.0, step=0.1)
            hspc = st.number_input("HSPC", value=3.75, min_value=0.0, max_value=100.0, step=0.1)
            chol = st.number_input("CHOL", value=0.0, min_value=0.0, max_value=100.0, step=0.1)
            peg = st.number_input("PEG", value=1.25, min_value=0.0, max_value=100.0, step=0.1)
            tfr = st.number_input("TFR", value=1.0, min_value=0.0, max_value=100.0, step=0.1)
            frr = st.number_input("FRR", value=3.0, min_value=0.0, max_value=100.0, step=0.1)
            buffer = st.selectbox("BUFFER", ["PBS", "MQ"])

            if st.button("Predict"):
                input_data = pd.DataFrame({
                    "ML": [ml],
                    "CHIP": [chip],
                    "ESM": [esm],
                    "HSPC": [hspc],
                    "CHOL": [chol],
                    "PEG": [peg],
                    "TFR": [tfr],
                    "FRR": [frr],
                    "BUFFER": [buffer],
                })
                predictions = model.predict(input_data)
                size, pdi = predictions[0]
                st.subheader("Model Predictions")
                st.write(f"`SIZE`: {size:.2f}")
                st.write(f"`PDI`: {pdi:.2f}")
        except FileNotFoundError:
            st.error(f"Model file not found: {MODEL_PATH}")

    # XGBoost Model
    elif option == "XGBoost":
        st.header("XGBoost")
        st.write("eXtreme Gradient Boosting predictions for `SIZE` and `PDI`.")
        MODEL_PATH = "models/xgboost_model.pkl"
        try:
            with open(MODEL_PATH, "rb") as file:
                model = pickle.load(file)
            st.write(f"Model loaded successfully from {MODEL_PATH}")

            # Input fields for prediction
            ml = st.selectbox("ML", ["HSPC", "ESM"])
            chip = st.selectbox("CHIP", ["Micromixer", "Droplet junction"])
            esm = st.number_input("ESM", value=0.0, min_value=0.0, max_value=100.0, step=0.1)
            hspc = st.number_input("HSPC", value=3.75, min_value=0.0, max_value=100.0, step=0.1)
            chol = st.number_input("CHOL", value=0.0, min_value=0.0, max_value=100.0, step=0.1)
            peg = st.number_input("PEG", value=1.25, min_value=0.0, max_value=100.0, step=0.1)
            tfr = st.number_input("TFR", value=1.0, min_value=0.0, max_value=100.0, step=0.1)
            frr = st.number_input("FRR", value=3.0, min_value=0.0, max_value=100.0, step=0.1)
            buffer = st.selectbox("BUFFER", ["PBS", "MQ"])

            if st.button("Predict"):
                input_data = pd.DataFrame({
                    "ML": [ml],
                    "CHIP": [chip],
                    "ESM": [esm],
                    "HSPC": [hspc],
                    "CHOL": [chol],
                    "PEG": [peg],
                    "TFR": [tfr],
                    "FRR": [frr],
                    "BUFFER": [buffer],
                })
                predictions = model.predict(input_data)
                size, pdi = predictions[0]
                st.subheader("Model Predictions")
                st.write(f"`SIZE`: {size:.2f}")
                st.write(f"`PDI`: {pdi:.2f}")
        except FileNotFoundError:
            st.error(f"Model file not found: {MODEL_PATH}")

    # Inverse Problem
    elif option == "Inverse problem":
        st.header("Inverse Problem")
        st.write("Given target `SIZE` and `PDI`, returns predictions for other numerical features.")
        MODEL_PATH = "models/inverse_xgboost_model.pkl"
        try:
            with open(MODEL_PATH, "rb") as file:
                model = pickle.load(file)
            st.write(f"Model loaded successfully from {MODEL_PATH}")

            size = st.number_input("SIZE", value=118.0, min_value=0.0, max_value=500.0, step=0.1)
            pdi = st.number_input("PDI", value=0.33, min_value=0.0, max_value=1.0, step=0.01)

            if st.button("Predict"):
                input_data = pd.DataFrame({"SIZE": [size], "PDI": [pdi]})
                predictions = model.predict(input_data)
                predictions = np.abs(predictions)
                predictions = np.where(predictions < 0.5, 0, predictions)
                st.subheader("Model Predictions")
                prediction_df = pd.DataFrame(predictions, columns=["ESM", "HSPC", "CHOL", "PEG", "TFR", "FRR"])
                st.write(prediction_df)
        except FileNotFoundError:
            st.error(f"Model file not found: {MODEL_PATH}")

# Visualization section
elif section == "Visualization":
    st.sidebar.title("Visualization Options")
    option = st.sidebar.selectbox(
        "Choose a Visualization:",
        [
            "Correlation heatmaps",
            "Alluvial plot",
            "Feature importance",
        ],
    )

    # Correlation Heatmap
    if option == "Correlation heatmaps":
        st.header("Correlation Heatmap")
        st.write("Displays the correlation between numerical features in the dataset.")
        if 'data' in locals():
            numeric_cols = data.select_dtypes(include=['float64', 'int64'])
            correlation_matrix = numeric_cols.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
            st.pyplot(plt)
        else:
            st.warning("No dataset available. Please upload your data.")

    # Placeholder for Alluvial Plot
    elif option == "Alluvial plot":
        st.header("Alluvial Plot")
        st.write("Visualize flows in the data.")
        st.info("Alluvial plot functionality is under development.")

    # Placeholder for Feature Importance
    elif option == "Feature importance":
        st.header("Feature Importance")
        st.write("Visualize the importance of features for predictions.")
        st.info("Feature importance visualization is under development.")
