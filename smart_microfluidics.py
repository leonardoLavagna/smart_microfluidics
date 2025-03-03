import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joypy
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
st.title("Smart Microfluidics: Machine Learning tools for Liposome Production Experiments")
st.sidebar.title("Choose an Option")
section = st.sidebar.selectbox(
    "Data preprocessing, data modelization or data visualization :",
    [
        "Dataset",
        "Data Modeling",
        "Data Exploration",
    ],
)


################################################
# 1. DATA
################################################
file_path = "data/data.csv"
data = pd.read_csv(file_path, encoding="latin1").drop(columns=['FR-O', 'FR-W'])
data.BUFFER = data.BUFFER.astype(str).str.strip().replace({'PBS\xa0': 'PBS'})
data.CHIP = data.CHIP.replace({'Micromixer\xa0': 'Micromixer'})
data = data.applymap(lambda x: str(x) if isinstance(x, str) else x)

if section == "Dataset":
    st.header("Dataset")
    st.write("Get the data for subsequent processing.")
    user_choice = st.radio("Upload custom data?", ("No", "Yes"))
    if user_choice == "Yes":
        st.warning("Custom data processing requires a premium account due to memory requirements, feature engineering and taylored processing.")
        uploaded_file = st.file_uploader("Drag and Drop your CSV file here", type=["csv"])
        if uploaded_file is not None:
            st.warning("Customized data processing not available for current user.")
            #df = pd.read_csv(uploaded_file)
            #st.success("File uploaded successfully!")
            #st.write("Preview of the uploaded data:")
            #st.dataframe(df)
                   
    else: 
        st.write("Loading default dataset...")
        #file_path = "data/cleaned_data.csv"
        #data = pd.read_csv(file_path, encoding="latin1").drop(columns=['FR-O', 'FR-W'])
        #data.BUFFER = data.BUFFER.astype(str).str.strip().replace({'PBS\xa0': 'PBS'})
        #data.CHIP = data.CHIP.replace({'Micromixer\xa0': 'Micromixer'})
        #data = data.applymap(lambda x: str(x) if isinstance(x, str) else x)
        data = pd.read_csv(file_path, encoding="latin1")
        st.success("File loaded successfully!")
        st.dataframe(data)


################################################
# 2.MODELS
################################################
if section == "Data Modeling":
    st.sidebar.title("Model Selection")
    option = st.sidebar.selectbox(
        "Choose a model:",
        [
            "Random forest regressor",
            "XGBoost",
            "Inverse problem",
            "Advanced models",
        ],
    )
        
    # 2.1 Random forest regressor
    if option == "Random forest regressor":
        st.header("Random forest regressor")
        st.markdown("Using multiple decision trees in parallel and bagging this model based on a [random forest](https://en.wikipedia.org/wiki/Random_forest) provides robust predictions for `SIZE` and `PDI`.")  
        with open(random_forest_model, "rb") as file:
            model = pickle.load(file)
        st.write(f"Loaded {random_forest_model} with the following performance metrics.")
        st.table(pd.DataFrame({
            "Metric": ["R-squared", "Mean Squared Error", "Mean Absolute Error"],
            "Value": [0.36157896454981364, 1958.890858993266, 15.086741645521377]
        }))
        st.write("Try the model with your data.")
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
        st.markdown("Using [eXtreme Gradient Boosting](https://en.wikipedia.org/wiki/XGBoost) the model provides joint predictions for `SIZE` and `PDI`.") 
        with open(xgboost_model, "rb") as file:
            model = pickle.load(file)
        st.write(f"Loaded {xgboost_model} with the following performance metrics.")
        st.table(pd.DataFrame({
            "Metric": ["R-squared", "Mean Squared Error", "Mean Absolute Error"],
            "Value": [0.32854801416397095, 1967.81201171875, 14.646432876586914]
        }))
        st.write("Try the model with your data.")
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
        st.markdown("This inference method pivots the pretrained XGBoost model to solve an [inverse problem](https://en.wikipedia.org/wiki/Inverse_problem): given target `SIZE` and `PDI` returns predictions for the other numerical features.")
        with open(inverse_xgboost_model, "rb") as file:
            model = pickle.load(file)
        st.write(f"Loaded {inverse_xgboost_model} with the folloing performance metrics.")
        st.table(pd.DataFrame({
            "Metric": ["R-squared", "Mean Squared Error", "Mean Absolute Error"],
            "Value": [0.11767681688070297, 89.26007080078125, 3.7459394931793213]
        }))
        st.write("Try the model with your data.")
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
        
    # 2.4 Advanced models
    elif option == "Advanced models":
        st.header("Advanced models")
        st.write("Taylored machine learning models for custom data.") 
        st.warning("The selected inference mode requires higher computational resources and customized architectures available to premium users only.")
        st.subheader("Preview of some available advanced models for predicting `SIZE` or `PDI`")
        st.write("`ensemble-pdi`")
        st.table(pd.DataFrame({
            "Metric": ["R-squared", "Mean Squared Error", "Mean Absolute Error"],
            "Value": [0.5328156582541348, 0.00245118805677467875, 0.04254484741522172]
        }))
        st.write("`ensemble-size`")
        st.table(pd.DataFrame({
            "Metric": ["R-squared", "Mean Squared Error", "Mean Absolute Error"],
            "Value": [0.874431019712665, 340.2617544655023, 11.834716059736861]
        }))
        st.write("With an improvement in terms of metrics given by the following plot.")
        metrics = ["R-squared", "MSE", "MAE"]
        before = [0.3285, 1967.81, 14.65]
        after = [0.8744, 340.26, 11.83]
        improvements = [
        ((after[i] - before[i]) / before[i]) * 100 if metrics[i] == "R-squared" 
        else ((before[i] - after[i]) / before[i]) * 100  
        for i in range(len(metrics))
        ]
        fig, ax1 = plt.subplots(figsize=(8, 5))
        bar_width = 0.35
        x = np.arange(len(metrics))
        bars_before = ax1.bar(x - bar_width/2, before, bar_width, label="Before", color="red", alpha=0.7)
        bars_after = ax1.bar(x + bar_width/2, after, bar_width, label="After", color="green", alpha=0.7)
        for bar in bars_before:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height():.2f}", 
                     ha="center", va="bottom", fontsize=10, color="black")
        for bar in bars_after:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height():.2f}", 
                     ha="center", va="bottom", fontsize=10, color="black")
        ax2 = ax1.twinx()
        ax2.plot(x, improvements, marker="o", linestyle="--", color="blue", label="Improvement (%)", linewidth=2)
        for i, val in enumerate(improvements):
            ax2.text(x[i], val, f"{val:.1f}%", ha="center", va="bottom", fontsize=12, color="blue")
        ax1.set_xlabel("Metrics")
        ax1.set_ylabel("Values (Before & After)")
        ax2.set_ylabel("Improvement (%)")
        ax1.set_title("Model Performance Improvement")
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend(loc="upper right", bbox_to_anchor=(1, 1), ncol=1, frameon=False)
        ax2.legend(loc="upper right", bbox_to_anchor=(1, 0.85), ncol=1, frameon=False)
        st.pyplot(fig)
        st.write("Depending on the number of samples available for training and validation we can boost the performances even further.")
        # 2.4.1 ensemble-size
        st.write("Try the `ensemble-size` model with your data.")
        with open(size_model, "rb") as file:
            model = pickle.load(file)
        tlp = st.number_input("TLP", value=5.0, min_value=0.0, max_value=100.0, step=0.1)
        esm = st.number_input("ESM", value=0.0, min_value=0.0, max_value=100.0, step=0.1)
        hspc = st.number_input("HSPC", value=3.75, min_value=0.0, max_value=100.0, step=0.1)
        chol = st.number_input("CHOL", value=0.0, min_value=0.0, max_value=100.0, step=0.1)
        peg = st.number_input("PEG", value=1.25, min_value=0.0, max_value=100.0, step=0.1)
        tfr = st.number_input("TFR", value=1.0, min_value=0.0, max_value=100.0, step=0.1)
        frr = st.number_input("FRR", value=3.0, min_value=0.0, max_value=100.0, step=0.1)
        if st.button("Predict"):
            input_data = pd.DataFrame({
                "TLP": [tlp],
                "ESM": [esm],
                "HSPC": [hspc],
                "CHOL": [chol],
                "PEG": [peg],
                "TFR ": [tfr],
                "FRR": [frr],
            })
            st.write(f"Predicted `SIZE`: {model.predict(input_data)}")


################################################
# 3. DATA EXPLORATION
################################################
elif section == "Data Exploration":
    st.sidebar.title("Visualization Options")
    option = st.sidebar.selectbox(
        "Choose a Visualization:",
        [
            "Ridgeline plot",
            "Correlation heatmaps",
            "Alluvial plot",
            "Feature importance",
            "PCA and clustering",
        ],
    )

    # 3.1 Ridgeline plot
    if option == "Ridgeline plot":
        st.header("Ridgeline plot")
        st.markdown("Displays the distributions of individual features as [overlapping density curves](https://en.wikipedia.org/wiki/Ridgeline_plot).")  
        numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
        df_numeric = data[numerical_cols]
        st.subheader("Numerical columns summary")
        st.write(df_numeric.describe())
        scaler = StandardScaler()
        df_standardized = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)
        st.subheader("Stacked densities")
        plt.figure(figsize=(12, 8))
        joypy.joyplot(df_standardized, colormap=plt.cm.coolwarm, x_range=[-5, 10], figsize=(12, 8))
        st.pyplot(plt)

    # 3.2 Correlation heatmap
    if option == "Correlation heatmaps":
        st.header("Correlation heatmap")
        st.markdown("Displays the [correlation between numerical features](https://en.wikipedia.org/wiki/Heat_map) in the dataset.")
        formed = st.text_input("Are you interested in the dataset of formed liposomes? Answer YES (Y) or NO (N):", "Y")
        n_ids = st.number_input("Enter the number of formulations you desire to visualize:", min_value=9, max_value=15, value=10)
        if formed == "YES" or formed == "Y":
            # 3.2.1 Correlations by features
            st.subheader("Correlations by features")
            numeric_cols = data.select_dtypes(include=['number']).columns
            yes_data = data[data['OUTPUT'] == 'YES']
            numeric_yes_data = yes_data[numeric_cols]
            correlation_matrix = numeric_yes_data.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
            st.pyplot(plt)
            # 3.2.2 Correlations by IDs
            st.subheader("Correlations by IDs")
            data_red = data[:n_ids]
            yes_data_red = data_red[data_red['OUTPUT'] == 'YES']
            numeric_yes_data_red = yes_data_red[numeric_cols]
            correlation_matrix_t = numeric_yes_data_red.T.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix_t, annot=True, fmt=".2f", cmap="coolwarm")
            st.pyplot(plt)
            # 3.2.3 Clustered correlations by features
            st.subheader("Clustered correlations by features")
            linkage_matrix = sch.linkage(numeric_yes_data, method='ward')
            sns.clustermap(numeric_yes_data.corr(),method='ward',cmap='coolwarm',annot=True,figsize=(10, 8))
            st.pyplot(plt)
            # 3.2.4 Clustered correlations by IDs
            st.subheader("Clustered correlations by IDs")
            linkage_matrix_red = sch.linkage(numeric_yes_data_red.T, method='ward')
            sns.clustermap(numeric_yes_data_red.T.corr(),method='ward',cmap='coolwarm',annot=True,figsize=(10, 8))
            st.pyplot(plt)
        else:
            # 3.2.5 Correlations by IDs
            numeric_cols = data.select_dtypes(include=['number']).columns
            no_data = data[data['OUTPUT'] == 'NO']
            numeric_no_data = no_data[numeric_cols]
            correlation_matrix = numeric_no_data.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
            st.pyplot(plt)
            # 3.2.6 Clustered correlations by IDs
            st.subheader("Clustered correlations by IDs")
            data_red = data[:n_ids]
            no_data_red = data_red[data_red['OUTPUT'] == 'NO']
            numeric_no_data_red = no_data_red[numeric_cols]
            correlation_matrix_t = numeric_no_data_red.T.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix_t, annot=True, fmt=".2f", cmap="coolwarm")
            st.pyplot(plt)
                
    # 3.3 Alluvial plot
    elif option == "Alluvial plot":
        st.header("Alluvial plot")
        st.markdown("Displays the [flow between categorical data](https://en.wikipedia.org/wiki/Alluvial_diagram).")
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = data[col].astype(str)
                for col in data.columns:
                    if data[col].dtype == 'object':
                        data[col] = data[col].astype(str)
        data['SIZE'] = data['SIZE'].apply(categorize_size)
        data['PDI'] = data['PDI'].apply(categorize_pdi)
        # 3.3.1 Sankey diagram  of two variables
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
          # 3.3.2 Sankey diagram of the ML->CHIP->BUFFER->OUTPUT flow
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
    
    # 3.4 Feature importance
    elif option == "Feature importance":
        st.header("Feature importance with a Random Forest Regressor")
        st.markdown("Displays the [importance of each feature](https://en.wikipedia.org/wiki/Feature_selection) for predicting a set of input targets.")
        # 3.4.1 Single target feature importance
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
        # 3.4.2 Two targets feature importance
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
            plt.figure(figsize=(10, 8))
            sns.barplot(data=joint_correlation_df, x='Mean Correlation', y='Feature', palette='viridis', dodge=False)
            plt.xlabel("Mean Correlation with Targets")
            plt.ylabel("Feature")
            plt.title("Feature Importance for Selected Targets")
            st.pyplot(plt)
    
    # 3.5 PCA and clustering
    elif option == "PCA and clustering":
        st.header("Principal component analysis and associated clusters.")
        st.markdown("Displays the principal data features and their clusters using a standard [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) algorithm, a [UMAP](https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction#Uniform_manifold_approximation_and_projection), and [k-means](https://en.wikipedia.org/wiki/K-means_clustering) clustering.")
        numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
        data_numeric = data[numerical_cols]
        st.write("Numerical columns summary")
        st.write(data_numeric.describe())
        #3.5.1 Standard PCA
        st.subheader("Standard PCA onto two principal components")
        scaler = StandardScaler()
        data_standardized = scaler.fit_transform(data_numeric)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data_standardized)
        df_pca = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
        plt.figure(figsize=(8, 6))
        sns.kdeplot(x=df_pca["PC1"], y=df_pca["PC2"], fill=True, cmap="Blues", thresh=0.05)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("2D Density Plot of Data Distribution (PCA Transformed)")
        st.pyplot(plt)
        #3.5.2 UMAP
        st.subheader("UMAP decomposition and k-means clustering")
        scaler = StandardScaler()
        data_standardized = scaler.fit_transform(data_numeric)
        n_components = st.slider("Select number of UMAP components:", min_value=2, max_value=10, value=2)
        umap_model = umap.UMAP(n_components=n_components, random_state=42)
        umap_result = umap_model.fit_transform(data_standardized)
        n_clusters = st.slider("Select number of clusters for K-Means:", min_value=2, max_value=10, value=4)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(umap_result)
        df_umap = pd.DataFrame(umap_result, columns=[f"UMAP{i+1}" for i in range(n_components)])
        df_umap["Cluster"] = clusters  # Add cluster labels
        st.write(f"UMAP Projection with K-Means Cluster Density (UMAP components: {n_components}, Clusters: {n_clusters}):")
        plt.figure(figsize=(10, 8))
        sns.kdeplot(x=df_umap[f"UMAP1"], y=df_umap[f"UMAP2"], hue=df_umap["Cluster"], fill=True, palette="tab10", thresh=0.05, alpha=0.7)
        plt.xlabel(f"UMAP Component 1")
        plt.ylabel(f"UMAP Component 2")
        plt.title(f"UMAP Projection with K-Means Cluster Density (Clusters: {n_clusters})")
        st.pyplot(plt)


