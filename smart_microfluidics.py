import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch
import pickle
from sklearn.ensemble import RandomForestRegressor
from config import *


################################################
# AUXILIARY FUNCTIONS
################################################  
def categorize_size(size):
    if size < 100:
        return 'S'
    elif size < 200:
        return 'M'
    else:
        return 'L'

def categorize_pdi(pdi):
    if pdi < 0.1:
        return 'HMD'
    elif 0.1 < pdi < 0.25:
        return 'MD'
    else:
        return 'PLD'



################################################
# COLOPHON
################################################                    
st.title("Smart Microfluidics: Modeling and Visualization")
st.sidebar.title("Choose an Option")
section = st.sidebar.selectbox(
    "Data preprocessing, data modelization or data visualization :",
    [
        "Dataset",
        "Modeling",
        "Visualization",
    ],
)


################################################
# 1. DATA
################################################

if section == "Dataset":
    st.header("Dataset")
    st.write("Get the data for subsequent processing.")
    user_choice = st.radio("Upload custom data?", ("No", "Yes"))
    if user_choice == "Yes":
        uploaded_file = st.file_uploader("Drag and Drop your CSV file here", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            st.write("Preview of the uploaded data:")
            st.dataframe(df)
        else:
            st.warning("No file uploaded. Please upload a CSV file.")
    else: 
        st.write("Loading default dataset...")
        #file_path = "data/cleaned_data.csv"
        #data = pd.read_csv(file_path, encoding="latin1").drop(columns=['FR-O', 'FR-W'])
        #data.BUFFER = data.BUFFER.astype(str).str.strip().replace({'PBS\xa0': 'PBS'})
        #data.CHIP = data.CHIP.replace({'Micromixer\xa0': 'Micromixer'})
        #data = data.applymap(lambda x: str(x) if isinstance(x, str) else x)
        data = pd.read_csv(file_path, encoding="latin1")
        st.dataframe(data)


################################################
# 2.MODELS
################################################
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
        
    # 2.1 Random forest regressor
    if option == "Random forest regressor":
        st.header("Random forest regressor")
        st.write("Using multiple decision trees in parallel and bagging this model provide robust predictions for `SIZE` and `PDI`.")  
        with open(random_forest_model, "rb") as file:
            model = pickle.load(file)
        st.write(f"Loaded {random_forest_model}")
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
                "TLP": [tlp],
                "ESM": [esm],
                "HSPC": [hspc],
                "CHOL": [chol],
                "PEG": [peg],
                "TFR ": [tfr],
                "FRR": [frr],
                "BUFFER": [buffer],
            })
            predictions = model.predict(input_data)
            size, pdi = predictions[0]
            st.subheader("Model predictions")
            st.write(f"`SIZE`: {size:.2f}")
            st.write(f"`PDI`: {pdi:.2f}")
        
    # 2.2 XGBoost
    elif option == "XGBoost":
        st.header("XGBoost")
        st.write("eXtreme Gradient Boosting predictions for `SIZE` and `PDI`.") 
        with open(xgboost_model, "rb") as file:
            model = pickle.load(file)
        st.write(f"Loaded {xgboost_model}")
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
                "TLP": [tlp],
                "ESM": [esm],
                "HSPC": [hspc],
                "CHOL": [chol],
                "PEG": [peg],
                "TFR ": [tfr],
                "FRR": [frr],
                "BUFFER": [buffer],
                "OUTPUT": [1]
            })
            predictions = model.predict(input_data)
            size, pdi = predictions[0]
            st.subheader("Model predictions")
            st.write(f"`SIZE`: {size:.2f}")
            st.write(f"`PDI`: {pdi:.2f}")
    
    # 2.3 Inverse model
    elif option == "Inverse problem":
        st.header("Inverse problem")
        st.write("Inverse problem solver: given target `SIZE` and `PDI` returns predictions for the other numerical features.")
        with open(inverse_xgboost_model, "rb") as file:
            model = pickle.load(file)
        st.write(f"Loaded {inverse_xgboost_model}")
        size = st.number_input("SIZE", value=118.0, min_value=0.0, max_value=500.0, step=0.1)
        pdi = st.number_input("PDI", value=0.33, min_value=0.0, max_value=1.0, step=0.01)
        if st.button("Predict"):
            input_data = pd.DataFrame({"SIZE": [size],"PDI": [pdi]})
            predictions = model.predict(input_data)
            predictions = np.abs(predictions)
            predictions = np.where(predictions < 0.5, 0, predictions)
            st.subheader("Model predictions")
            prediction_df = pd.DataFrame(predictions, columns=["ESM", "HSPC", "CHOL", "PEG", "TFR", "FRR"])
            st.write(prediction_df)

        
################################################
# 3. VISUALIZATIONS
################################################
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
    file_path = "data/data.csv"
    data = pd.read_csv(file_path, encoding="latin1").drop(columns=['FR-O', 'FR-W'])
    data.BUFFER = data.BUFFER.astype(str).str.strip().replace({'PBS\xa0': 'PBS'})
    data.CHIP = data.CHIP.replace({'Micromixer\xa0': 'Micromixer'})
    data = data.applymap(lambda x: str(x) if isinstance(x, str) else x)

    # 3.1 Correlation heatmap
    if option == "Correlation heatmaps":
        st.header("Correlation heatmap")
        st.write("Displays the correlation between numerical features in the dataset.")
        formed = st.text_input("Are you interested in the dataset of formed liposomes? Answer YES (Y) or NO (N):", "Y")
        n_ids = st.number_input("Enter the number of formulations you desire to visualize:", min_value=9, max_value=15, value=10)
        if formed == "YES" or formed == "Y":
            # 3.1.1 Correlations by features
            st.subheader("Correlations by features")
            numeric_cols = data.select_dtypes(include=['number']).columns
            yes_data = data[data['OUTPUT'] == 'YES']
            numeric_yes_data = yes_data[numeric_cols]
            correlation_matrix = numeric_yes_data.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
            st.pyplot(plt)
            # 3.1.2 Correlations by IDs
            st.subheader("Correlations by IDs")
            data_red = data[:n_ids]
            yes_data_red = data_red[data_red['OUTPUT'] == 'YES']
            numeric_yes_data_red = yes_data_red[numeric_cols]
            correlation_matrix_t = numeric_yes_data_red.T.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix_t, annot=True, fmt=".2f", cmap="coolwarm")
            st.pyplot(plt)
            # 3.1.3 Clustered correlations by features
            st.subheader("Clustered correlations by features")
            linkage_matrix = sch.linkage(numeric_yes_data, method='ward')
            sns.clustermap(numeric_yes_data.corr(),method='ward',cmap='coolwarm',annot=True,figsize=(10, 8))
            st.pyplot(plt)
            # 3.1.4 Clustered correlations by IDs
            st.subheader("Clustered correlations by IDs")
            linkage_matrix_red = sch.linkage(numeric_yes_data_red.T, method='ward')
            sns.clustermap(numeric_yes_data_red.T.corr(),method='ward',cmap='coolwarm',annot=True,figsize=(10, 8))
            st.pyplot(plt)
        else:
            # 3.1.5 Correlations by IDs
            numeric_cols = data.select_dtypes(include=['number']).columns
            no_data = data[data['OUTPUT'] == 'NO']
            numeric_no_data = no_data[numeric_cols]
            correlation_matrix = numeric_no_data.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
            st.pyplot(plt)
            # 3.1.6 Clustered correlations by IDs
            st.subheader("Clustered correlations by IDs")
            data_red = data[:n_ids]
            no_data_red = data_red[data_red['OUTPUT'] == 'NO']
            numeric_no_data_red = no_data_red[numeric_cols]
            correlation_matrix_t = numeric_no_data_red.T.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix_t, annot=True, fmt=".2f", cmap="coolwarm")
            st.pyplot(plt)
                
    # 3.2 Alluvial plot
    elif option == "Alluvial plot":
        st.header("Alluvial plot")
        st.write("Displays the flow of categorical data using an alluvial plot.")
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = data[col].astype(str)
                for col in data.columns:
                    if data[col].dtype == 'object':
                        data[col] = data[col].astype(str)
        data['SIZE'] = data['SIZE'].apply(categorize_size)
        data['PDI'] = data['PDI'].apply(categorize_pdi)
        # 3.2.1 Sankey diagram  of two variables
        st.subheader("Sankey diagram of two variables")
        source = st.selectbox("Choose the first categorical variable of interest.", ("ML", "CHIP", "BUFFER", "OUTPUT", "SIZE", "PDI"))
        target = st.selectbox("Choose the second categorical variable of interest (different from the first).", ("CHIP", "ML", "BUFFER", "OUTPUT", "SIZE", "PDI"))
        if source == target:
            st.write(":red[INPUT ERROR. The selected variables cannot be equal.]")
        elif source != target:
          value_counts = data.groupby([source, target]).size().reset_index(name='value')
          categories = list(set(value_counts[source]).union(set(value_counts[target])))
          category_to_index = {category: i for i, category in enumerate(categories)}
          sources = value_counts[source].map(category_to_index).tolist()
          targets = value_counts[target].map(category_to_index).tolist()
          values = value_counts['value'].tolist()
          sankey_fig = go.Figure(data=[go.Sankey(node=dict(pad=15,thickness=20,line=dict(color="black", width=0.5),label=categories),
                                                 link=dict(source=sources,target=targets,value=values))])
          st.plotly_chart(sankey_fig)
          # 3.2.2 Sankey diagram of the ML->CHIP->BUFFER->OUTPUT flow
          st.subheader("Sankey diagram of the ML->CHIP->BUFFER->OUTPUT flow")
          value_counts = data.groupby(['ML', 'CHIP', 'BUFFER', 'OUTPUT']).size().reset_index(name='value')
          all_categories = []
          for col in ['ML', 'CHIP', 'BUFFER', 'OUTPUT']:
              all_categories.extend(value_counts[col].unique())
          all_categories = list(set(all_categories))  # Remove duplicates
          category_to_index = {category: index for index, category in enumerate(all_categories)}
          sources = []
          targets = []
          values = []
          for index, row in value_counts.iterrows():
              sources.append(category_to_index[row['ML']])
              targets.append(category_to_index[row['CHIP']])
              values.append(row['value'])
              sources.append(category_to_index[row['CHIP']])
              targets.append(category_to_index[row['BUFFER']])
              values.append(row['value'])
              sources.append(category_to_index[row['BUFFER']])
              targets.append(category_to_index[row['OUTPUT']])
              values.append(row['value'])
          fig = go.Figure(data=[go.Sankey(node=dict(pad=15,thickness=20,line=dict(color="black", width=0.5),label=all_categories),
                                        link=dict(source=sources,target=targets,value=values))])
          st.plotly_chart(fig)
    
    # 3.3 Feature importance
    elif option == "Feature importance":
        st.header("Feature importance with a Random Forest Regressor")
        st.write("Displays the importance of each feature for predicting a set of input targets.")
        # 3.3.1 Single target feature importance
        st.subheader("Single target feature importance")
        target_feature = st.selectbox("Select a target feature.", ("TLP", "ESM", "HSPC", "CHOL", "PEG", "FRR", "SIZE", "PDI"))
        numeric_data = data.select_dtypes(include=['float64', 'int64']).dropna()
        if target_feature == "SIZE":
            numeric_data = numeric_data.drop(columns=["PDI"])
        if target_feature == "PDI":
            numeric_data = numeric_data.drop(columns=["SIZE"])        
        X = numeric_data.drop(columns=[target_feature])
        if "SIZE" in X.columns:
            X = X.drop(columns=["SIZE"])
        if "PDI" in X.columns:
            X = X.drop(columns=["PDI"])
        y = numeric_data[target_feature]  
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        feature_importances = pd.DataFrame({'Feature': X.columns,'Importance': rf.feature_importances_}).sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importances, x='Importance', y='Feature', hue='Feature', palette='viridis', dodge=False)
        st.pyplot(plt)
        # 3.3.2 Two targets feature importance
        available_features = numeric_data.columns.tolist()
        target_feature_1 = st.selectbox("Select the first target feature.", available_features)
        target_feature_2 = st.selectbox("Select the second target feature (different from the first).", available_features)
        if target_feature_1 == target_feature_2:
            st.error("INPUT ERROR: The selected variables cannot be the same.")
        elif target_feature_1 != target_feature_2:
            correlation_matrix = numeric_data.corr()
            filtered_data = numeric_data.copy()
            if target_feature_1 == "SIZE" or target_feature_2 == "SIZE":
                filtered_data = filtered_data.drop(columns=["PDI"], errors='ignore')
            if target_feature_1 == "PDI" or target_feature_2 == "PDI":
                filtered_data = filtered_data.drop(columns=["SIZE"], errors='ignore')
            target_features = [target_feature_1, target_feature_2]
            joint_correlation = correlation_matrix[target_features].drop(index=target_features, errors='ignore').mean(axis=1)
            joint_correlation_df = joint_correlation.reset_index()
            joint_correlation_df.columns = ['Feature', 'Mean Correlation']
            joint_correlation_df = joint_correlation_df.sort_values(by='Mean Correlation', ascending=False)
            plt.figure(figsize=(10, 6))
            sns.barplot(data=joint_correlation_df, x='Mean Correlation', y='Feature', palette='viridis', dodge=False)
            plt.xlabel("Mean Correlation with Targets")
            plt.ylabel("Feature")
            plt.title("Feature Importance for Selected Targets")
            st.pyplot(plt)
