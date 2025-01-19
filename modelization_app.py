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

# App title
st.title("Regression Model Training, Prediction, and Visualization")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    # Load data
    data = pd.read_csv(uploaded_file, encoding='latin1')
    data = data.set_index('ID')
    st.write("Dataset Preview:", data.head())

    # Define feature and target columns
    numerical_features = ['TLP', 'CHOL', 'TFR', 'FRR']
    categorical_features = ['CHIP', 'MEDIUM']
    target_columns = ['SIZE', 'PDI']

    # Check if the required columns are in the dataset
    required_columns = numerical_features + categorical_features + target_columns
    if not all(col in data.columns for col in required_columns):
        st.error(f"The dataset must contain the following columns: {required_columns}")
    else:
        # Split features and targets
        X = data[numerical_features + categorical_features]
        y = data[target_columns]

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocessing
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        # Model definition
        base_model = RandomForestRegressor(random_state=42)
        model = MultiOutputRegressor(base_model)

        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        # Train model
        if st.button("Train Model"):
            pipeline.fit(X_train, y_train)
            st.success("Model trained successfully!")

            # Save the trained model
            with open("trained_model.pkl", "wb") as file:
                pickle.dump(pipeline, file)
            st.info("Model saved as 'trained_model.pkl'.")

        # Load the trained model
        if st.button("Load Saved Model"):
            try:
                with open("trained_model.pkl", "rb") as file:
                    saved_pipeline = pickle.load(file)
                st.success("Model loaded successfully!")
            except FileNotFoundError:
                st.error("No saved model found. Please train and save the model first.")

        # Make predictions
        st.subheader("Make Predictions")
        with st.form("prediction_form"):
            # Input fields for numerical features
            numerical_inputs = {}
            for feature in numerical_features:
                numerical_inputs[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

            # Input fields for categorical features
            categorical_inputs = {}
            for feature in categorical_features:
                categorical_inputs[feature] = st.selectbox(f"Select value for {feature}", options=data[feature].unique())

            # Submit button
            submitted = st.form_submit_button("Predict")
            if submitted:
                # Prepare input for prediction
                input_data = {**numerical_inputs, **categorical_inputs}
                input_df = pd.DataFrame([input_data])

                # Make prediction
                if 'saved_pipeline' in locals():
                    prediction = saved_pipeline.predict(input_df)
                    prediction_df = pd.DataFrame(prediction, columns=target_columns)
                    st.write("Predicted Values:")
                    st.write(prediction_df)

                    # Visualization Options
                    st.subheader("Visualize Prediction Results")
                    visualization_type = st.selectbox("Select a visualization type", ["Bar Chart", "Scatter Plot"])

                    if visualization_type == "Bar Chart":
                        fig, ax = plt.subplots(figsize=(8, 4))
                        prediction_df.plot(kind="bar", ax=ax, legend=False, color=['skyblue', 'orange'])
                        ax.set_title("Prediction Results (Bar Chart)")
                        ax.set_ylabel("Predicted Values")
                        ax.set_xticks(range(len(target_columns)))
                        ax.set_xticklabels(target_columns, rotation=0)
                        st.pyplot(fig)

                    elif visualization_type == "Scatter Plot":
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.scatter(target_columns, prediction[0], color='green')
                        ax.set_title("Prediction Results (Scatter Plot)")
                        ax.set_ylabel("Predicted Values")
                        ax.set_xlabel("Target Features")
                        for i, txt in enumerate(prediction[0]):
                            ax.annotate(f"{txt:.2f}", (target_columns[i], prediction[0][i]), ha='center', va='bottom')
                        st.pyplot(fig)
                else:
                    st.error("No trained model available. Please train or load a model first.")
