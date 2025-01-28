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
        
    # 1. Random forest regressor
    if option == "Random forest regressor":
        st.header("Random forest regressor")
        st.write("Using multiple decision trees in parallel and bagging this model provide robust predictions for `SIZE` and `PDI`.")
        MODEL_PATH = "models/random_forest_model.pkl"  
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)
        st.write(f"Loaded {MODEL_PATH}")
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
                "TFR ": [tfr],
                "FRR": [frr],
                "BUFFER": [buffer],
            })
            predictions = model.predict(input_data)
            size, pdi = predictions[0]
            st.subheader("Model predictions")
            st.write(f"`SIZE`: {size:.2f}")
            st.write(f"`PDI`: {pdi:.2f}")
        
    # 2. XGBoost
    elif option == "XGBoost":
        st.header("XGBoost")
        st.write("eXtreme Gradient Boosting predictions for `SIZE` and `PDI`.")
        MODEL_PATH = "models/xgboost_model.pkl"  
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)
        st.write(f"Loaded {MODEL_PATH}")
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
    
    # 3. Inverse model
    elif option == "Inverse problem":
        st.header("Inverse problem")
        st.write("Inverse problem solver: given target `SIZE` and `PDI` returns predictions for the other numerical features.")
        MODEL_PATH = "models/inverse_xgboost_model.pkl"  
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)
        st.write(f"Loaded {MODEL_PATH}")
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

    # Visualizations
    # 1. Correlation heatmap
    if option == "Correlation heatmaps":
        st.header("Correlation heatmap")
        st.write("Displays the correlation between numerical features in the dataset.")
        file_path = "data/data.csv"
        data = pd.read_csv(file_path, encoding="latin1")
        data = data.drop(columns=['FR-O', 'FR-W'])
        data.BUFFER = data.BUFFER.astype(str).str.strip()
        data.BUFFER = data.BUFFER.replace({'PBS\xa0': 'PBS'})
        data.CHIP = data.CHIP.replace({'Micromixer\xa0': 'Micromixer'})
        data.SIZE = data.SIZE.astype(float)
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = data[col].astype(str)
                for col in data.columns:
                    if data[col].dtype == 'object':
                        data[col] = data[col].astype(str)

        formed = st.text_input("Are you interested in the dataset of formed liposomes? Answer YES (Y) or NO (N):", "Y")
        n_ids = st.number_input("Enter the number of formulations you desire to visualize:", min_value=9, max_value=15, value=10)
        if formed == "YES" or formed == "Y":
            # 1.1(Y) Correlations by features
            st.subheader("Correlations by features")
            numeric_cols = data.select_dtypes(include=['number']).columns
            yes_data = data[data['OUTPUT'] == 'YES']
            numeric_yes_data = yes_data[numeric_cols]
            correlation_matrix = numeric_yes_data.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
            st.pyplot(plt)
            # 1.2(Y) Correlations by IDs
            st.subheader("Correlations by IDs")
            data_red = data[:n_ids]
            yes_data_red = data_red[data_red['OUTPUT'] == 'YES']
            numeric_yes_data_red = yes_data_red[numeric_cols]
            correlation_matrix_t = numeric_yes_data_red.T.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix_t, annot=True, fmt=".2f", cmap="coolwarm")
            st.pyplot(plt)
            # 1.3(Y) Clustered correlations by features
            st.subheader("Clustered correlations by features")
            linkage_matrix = sch.linkage(numeric_yes_data, method='ward')
            sns.clustermap(numeric_yes_data.corr(),method='ward',cmap='coolwarm',annot=True,figsize=(10, 8))
            st.pyplot(plt)
            # 1.2(Y) Clustered correlations by IDs
            st.subheader("Clustered correlations by IDs")
            linkage_matrix_red = sch.linkage(numeric_yes_data_red.T, method='ward')
            sns.clustermap(numeric_yes_data_red.T.corr(),method='ward',cmap='coolwarm',annot=True,figsize=(10, 8))
            st.pyplot(plt)
        else:
            # Select numeric columns for correlation calculation corresponding to no-formation
            numeric_cols = data.select_dtypes(include=['number']).columns
            no_data = data[data['OUTPUT'] == 'NO']
            numeric_no_data = no_data[numeric_cols]
            # Plot heatmap
            correlation_matrix = numeric_no_data.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
            st.pyplot(plt)        
            # Plot heatmap by IDs
            st.subheader("Correlations by IDs")
            data_red = data[:n_ids]
            no_data_red = data_red[data_red['OUTPUT'] == 'NO']
            numeric_no_data_red = no_data_red[numeric_cols]
            correlation_matrix_t = numeric_no_data_red.T.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix_t, annot=True, fmt=".2f", cmap="coolwarm")
            st.pyplot(plt)
    
    elif option == "Alluvial plot":
        st.header("Alluvial plot")
        st.write("Displays the flow of categorical data using an alluvial plot.")
        file_path = "data/data.csv"
        data = pd.read_csv(file_path, encoding="latin1")
        data = data.drop(columns=['FR-O', 'FR-W'])
        data.BUFFER = data.BUFFER.astype(str).str.strip()
        data.BUFFER = data.BUFFER.replace({'PBS\xa0': 'PBS'})
        data.CHIP = data.CHIP.replace({'Micromixer\xa0': 'Micromixer'})
        data.SIZE = data.SIZE.astype(float)
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = data[col].astype(str)
                for col in data.columns:
                    if data[col].dtype == 'object':
                        data[col] = data[col].astype(str)
        def categorize_size(size):
            if size < 100:
                return 'S'
            elif size < 200:
                return 'M'
            else:
                return 'L'
        data['SIZE'] = data['SIZE'].apply(categorize_size)
        def categorize_pdi(pdi):
            if pdi < 0.1:
                return 'HMD'
            elif 0.1 < pdi < 0.25:
                return 'MD'
            else:
                return 'PLD'
        data['PDI'] = data['PDI'].apply(categorize_pdi)
        # 2.1 Sankey diagram  of two variables
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
          # 2.2 Sankey diagram of the ML->CHIP->BUFFER->OUTPUT flow
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
    
    elif option == "Feature importance":
        st.header("Feature importance with a Random Forest Regressor")
        st.write("Displays the importance of each feature for predicting a set of input targets.")
        file_path = "data/data.csv"
        data = pd.read_csv(file_path, encoding="latin1")
        data = data.drop(columns=['FR-O', 'FR-W'])
        data.BUFFER = data.BUFFER.astype(str).str.strip()
        data.BUFFER = data.BUFFER.replace({'PBS\xa0': 'PBS'})
        data.CHIP = data.CHIP.replace({'Micromixer\xa0': 'Micromixer'})
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = data[col].astype(str)
                for col in data.columns:
                    if data[col].dtype == 'object':
                        data[col] = data[col].astype(str)
        # 3.1 Single target feature importance
        st.subheader("Single target feature importance")
        target_feature = st.selectbox("Select a target feature.", ("TLP", "ESM", "HSPC", "CHOL", "PEG", "FRR", "SIZE", "PDI"))
        numeric_data = data.select_dtypes(include=['float64', 'int64']).dropna()
        X = numeric_data.drop(columns=[target_feature]) 
        X = X.drop(columns=["SIZE","PDI"])
        y = numeric_data[target_feature]  
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        feature_importances = pd.DataFrame({'Feature': X.columns,'Importance': rf.feature_importances_}).sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importances, x='Importance', y='Feature', hue='Feature', palette='viridis', dodge=False, legend=False)
        st.pyplot(plt)
        # 3.2 Two targets feature importance
        st.subheader("Two targets feature importance")
        target_feature_1 = st.selectbox("Select the first target feature.", ("TLP", "ESM", "HSPC", "CHOL", "PEG", "FRR"))
        target_feature_2 = st.selectbox("Select the second target feature (different from the first).", ("TLP", "ESM", "HSPC", "CHOL", "PEG", "FRR"))
        if target_feature_1 == target_feature_2:
            st.write(":red[INPUT ERROR. The selected variables cannot be equal.]")
        elif target_feature_1 != target_feature_2:
            numeric_data = numeric_data.drop(columns=["SIZE","PDI"])
            correlation_matrix = numeric_data.corr()
            target_features = [target_feature_1, target_feature_2]
            joint_correlation = correlation_matrix[target_features].drop(index=target_features).mean(axis=1)
            joint_correlation_df = joint_correlation.reset_index()
            joint_correlation_df.columns = ['Feature', 'Mean Correlation']
            joint_correlation_df = joint_correlation_df.sort_values(by='Mean Correlation', ascending=False)
            plt.figure(figsize=(10, 6))
            sns.barplot(data=joint_correlation_df, x='Mean Correlation', y='Feature', hue='Feature', palette='viridis', dodge=False, legend=False)
            st.pyplot(plt)
