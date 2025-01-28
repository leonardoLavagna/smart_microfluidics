import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import plotly.express as px
from plotly import graph_objects as go
import scipy.cluster.hierarchy as sch
import pickle
from sklearn.ensemble import RandomForestRegressor

# Load data
uploaded_file = st.file_uploader("Drag and Drop your CSV file here", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        
        # Display the dataframe
        st.dataframe(df)
        
        # Optionally show more insights or a preview
        st.write("Preview of the file:")
        st.write(df.head())
    except Exception as e:
        st.error(f"Error reading the file: {e}")
else:
    st.info("Please upload a CSV file.")
    
file_path = 'data/data.csv'
data = pd.read_csv(file_path, encoding='latin1')
data = data.set_index('ID')

# Streamlit app title
st.title("Smart Microfluidics: Modeling and Visualization")

# Sidebar for navigation
st.sidebar.title("Choose an Option")
section = st.sidebar.selectbox(
    "Section:",
    [
        "Modeling",
        "Visualization",
    ],
)

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

    # Random forest regressor
    if option == "Random forest regressor":
        st.header("Random Forest Regressor")
        st.write("Using multiple decision trees in parallel for robust predictions of `SIZE` and `PDI`.")
        MODEL_PATH = "models/random_forest_model.pkl"
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)
        st.write(f"Loaded {MODEL_PATH}")

        # Input fields
        ml = st.selectbox("ML", ["HSPC", "ESM"])
        chip = st.selectbox("CHIP", ["Micromixer", "Droplet junction"])
        tlp = st.number_input("TLP", value=5.0, min_value=0.0, max_value=100.0, step=0.1)
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

    # XGBoost
    elif option == "XGBoost":
        st.header("XGBoost")
        st.write("eXtreme Gradient Boosting predictions for `SIZE` and `PDI`.")
        MODEL_PATH = "models/xgboost_model.pkl"
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)
        st.write(f"Loaded {MODEL_PATH}")

        # Input fields
        ml = st.selectbox("ML", ["HSPC", "ESM"])
        chip = st.selectbox("CHIP", ["Micromixer", "Droplet junction"])
        tlp = st.number_input("TLP", value=5.0, min_value=0.0, max_value=100.0, step=0.1)
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
                "OUTPUT": [1],
            })
            predictions = model.predict(input_data)
            size, pdi = predictions[0]
            st.subheader("Model Predictions")
            st.write(f"`SIZE`: {size:.2f}")
            st.write(f"`PDI`: {pdi:.2f}")

    # Inverse problem
    elif option == "Inverse problem":
        st.header("Inverse Problem")
        st.write("Given target `SIZE` and `PDI`, returns predictions for other numerical features.")
        MODEL_PATH = "models/inverse_xgboost_model.pkl"
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)
        st.write(f"Loaded {MODEL_PATH}")

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

    # Correlation heatmaps
    if option == "Correlation heatmaps":
        st.header("Correlation Heatmap")
        st.write("Displays the correlation between numerical features in the dataset.")
        formed = st.text_input("Are you interested in the dataset of formed liposomes? Answer YES (Y) or NO (N):", "Y")
        n_ids = st.number_input("Enter the number of formulations to visualize:", min_value=9, max_value=15, value=10)

        if formed in ["YES", "Y"]:
            yes_data = data[data['OUTPUT'] == 'YES']
            numeric_yes_data = yes_data.select_dtypes(include=['number'])
            correlation_matrix = numeric_yes_data.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
            st.pyplot(plt)
        else:
            no_data = data[data['OUTPUT'] == 'NO']
            numeric_no_data = no_data.select_dtypes(include=['number'])
            correlation_matrix = numeric_no_data.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
            st.pyplot(plt)

    # Alluvial plot
    elif option == "Alluvial plot":
        st.header("Alluvial Plot")
        source = st.selectbox("Choose the first categorical variable:", ["ML", "CHIP", "BUFFER", "OUTPUT"])
        target = st.selectbox("Choose the second categorical variable:", ["CHIP", "ML", "BUFFER", "OUTPUT"])

        if source == target:
            st.write(":red[INPUT ERROR: Variables cannot be the same.]")
        else:
            value_counts = data.groupby([source, target]).size().reset_index(name='value')
            categories = list(set(value_counts[source]).union(set(value_counts[target])))
            category_to_index = {category: i for i, category in enumerate(categories)}
            sources = value_counts[source].map(category_to_index).tolist()
            targets = value_counts[target].map(category_to_index).tolist()
            values = value_counts['value'].tolist()

            sankey_fig = go.Figure(data=[go.Sankey(
                node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=categories),
                link=dict(source=sources, target=targets, value=values))
            ])
            st.plotly_chart(sankey_fig)

    # Feature importance
    elif option == "Feature importance":
        st.header("Feature Importance")
        target_feature = st.selectbox("Select a target feature:", data.columns)
        numeric_data = data.select_dtypes(include=['number']).dropna()

        X = numeric_data.drop(columns=[target_feature])
        y = numeric_data[target_feature]
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)

        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importances, x='Importance', y='Feature', palette='viridis')
        st.pyplot(plt)
