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


@st.cache_data
def fetch_data(file_path):
    data = pd.read_csv(file_path, encoding="latin1").drop(columns=['ID', 'FR-O', 'FR-W'])
    data.BUFFER = data.BUFFER.astype(str).str.strip().replace({'PBS\xa0': 'PBS'})
    data.CHIP = data.CHIP.replace({'Micromixer\xa0': 'Micromixer'})
    data = data.map(lambda x: str(x) if isinstance(x, str) else x)
    numeric_data = data.select_dtypes(include=['float64', 'int64']).dropna()
    return data, numeric_data
    

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
data, numeric_data = fetch_data(file_path)
if section == "Dataset":
    st.markdown("""This app provides machine learning and data analysis tools for laboratory operators while carrying out microfluidic liposome experiments.
               The app is in a developing phase. Current version: `v0.2`. In this version the user can:""")
    st.markdown("""- Work with pretrained machine learning architectures on a default dataset;""")
    st.markdown("""- Simulate experiments with the tools in the Data Modeling section (menù on the left); """)
    st.markdown("""- Get in depth data analysis with the visualizations in the Data Exploration section (menù on the left). """)
    st.markdown("""Report bugs or request updates [@Leonardo Lavagna](https://leonardolavagna.github.io/) by [opening an issue in the project repository](https://github.com/leonardoLavagna/smart_microfluidics/issues). 
                The training of the models was done on a publicly available dataset and in compliance with current legislation.""")
    st.header("Dataset")
    st.warning(":warning: Current models have been trained on the default dataset which is of intermediate dimension. Higher performances require additional training data.")
    st.write("Get the data for subsequent processing.")
    user_choice = st.radio("Upload custom data?", ("No", "Yes"))
    if user_choice == "Yes":
        #st.warning("Custom data processing requires a premium account due to memory requirements, feature engineering and taylored processing.")
        st.warning(":no_entry_sign: Customized data processing not available for current user.") 
        #uploaded_file = st.file_uploader("Drag and Drop your CSV file here", type=["csv"])
        #if uploaded_file is not None:
        #    st.warning("Customized data processing not available for current user.")
            #df = pd.read_csv(uploaded_file)
            #st.success("File uploaded successfully!")
            #st.write("Preview of the uploaded data:")
            #st.dataframe(df)
                   
    else: 
        st.write("Loading default dataset...")
        st.success("File loaded successfully!")
        st.dataframe(data)
        #st.dataframe(data.style.format(thousands=""))
        

################################################
# 2.MODELS
################################################
if section == "Data Modeling":
    st.write("Start by choosing a model, then try the selected model with your data.")
    st.write("**Choose a model.**")
    option = st.radio(
        "",
        [
            "Random forest regressor",
            "XGBoost",
            "Inverse model",
            "Advanced models",
        ],
    )
    st.write("**Input your data.**")
    if option != "Inverse model":
        ml = st.selectbox("ML", ["HSPC", "ESM"])
        chip = st.selectbox("CHIP", ["Micromixer", "Droplet junction"])
        #tlp = st.number_input("TLP", value=5.0, min_value=0.0, max_value=100.0, step=0.1)
        esm_disabled = ml == "HSPC"
        hspc_disabled = ml == "ESM"
        esm = st.number_input("ESM", value=0.0 if esm_disabled else 0.1, min_value=0.0, max_value=100.0, step=0.1, disabled=esm_disabled)
        hspc = st.number_input("HSPC", value=0.0 if hspc_disabled else 3.75, min_value=0.0, max_value=100.0, step=0.1, disabled=hspc_disabled)
        chol = st.number_input("CHOL", value=0.0, min_value=0.0, max_value=100.0, step=0.1)
        peg = st.number_input("PEG", value=1.25, min_value=0.0, max_value=100.0, step=0.1)
        tfr = st.number_input("TFR", value=1.0, min_value=0.0, max_value=100.0, step=0.1)
        frr = st.number_input("FRR", value=3.0, min_value=0.0, max_value=100.0, step=0.1)
        buffer = st.selectbox("BUFFER", ["PBS", "MQ"])
        tlp = (hspc if hspc > 0 else esm) + chol + peg
    else:
        size = st.number_input("SIZE", value=118.0, min_value=0.0, max_value=500.0, step=0.1)
        pdi = st.number_input("PDI", value=0.33, min_value=0.0, max_value=1.0, step=0.01)
    
    # 2.1 Random forest regressor
    if option == "Random forest regressor":
        st.header("Random forest regressor")
        st.markdown("Using multiple [decision trees](https://en.wikipedia.org/wiki/Decision_tree) in parallel and [bagging](http://en.wikipedia.org/wiki/Bootstrap_aggregating) this model based on a [random forest](https://en.wikipedia.org/wiki/Random_forest) provides joint predictions for `SIZE` and `PDI`.")  
        with open(random_forest_model, "rb") as file:
            model = pickle.load(file)
        st.write(f"Loaded {random_forest_model} with the following performance metrics. The key metric is the square root of the error (either the MSE or the MAE).")
        st.table(pd.DataFrame({
            "Metric": ["R-squared", "Mean Squared Error (MSE)", "Mean Absolute Error (MAE)"],
            "Value": [0.36157896454981364, 1958.890858993266, 15.086741645521377]
        }))
        if st.button("Predict"):
            input_data = pd.DataFrame({
                "ML": [ml],
                "CHIP": [chip],
                "TLP": [tlp],
                "ESM": [0.0 if esm_disabled else esm],  
                "HSPC": [0.0 if hspc_disabled else hspc],  
                "CHOL": [chol],
                "PEG": [peg],
                "TFR ": [tfr],
                "FRR": [frr],
                "BUFFER": [buffer],
            })
            st.markdown("**Input data**")
            st.write(input_data)
            predictions = model.predict(input_data)
            size, pdi = predictions[0]
            st.markdown("**Model predictions**")
            if size > 500 or pdi > 0.5:
                st.write("The sistem doesn't form")
                st.write(f"`OUTPUT`: 0")
            else:
                st.write(f"`SIZE`: {size:.2f}")
                st.write(f"`PDI`: {pdi:.2f}")
               
    # 2.2 XGBoost
    elif option == "XGBoost":
        st.header("XGBoost")
        st.markdown("Using [eXtreme Gradient Boosting](https://en.wikipedia.org/wiki/XGBoost) the model provides joint predictions for `SIZE` and `PDI`.") 
        with open(xgboost_model, "rb") as file:
            model = pickle.load(file)
        st.write(f"Loaded {xgboost_model} with the following performance metrics. The key metric is the square root of the error (either the MSE or the MAE).")
        st.table(pd.DataFrame({
            "Metric": ["R-squared", "Mean Squared Error (MSE)", "Mean Absolute Error (MAE)"],
            "Value": [0.32854801416397095, 1967.81201171875, 14.646432876586914]
        }))
        if st.button("Predict"):
            input_data = pd.DataFrame({
                "ML": [ml],
                "CHIP": [chip],
                "TLP": [tlp],
                "ESM": [0.0 if esm_disabled else esm],  
                "HSPC": [0.0 if hspc_disabled else hspc],  
                "CHOL": [chol],
                "PEG": [peg],
                "TFR ": [tfr],
                "FRR": [frr],
                "BUFFER": [buffer],
            })
            input_data_ = pd.DataFrame({
                "ML": [ml],
                "CHIP": [chip],
                "TLP": [tlp],
                "ESM": [0.0 if esm_disabled else esm],  
                "HSPC": [0.0 if hspc_disabled else hspc],  
                "CHOL": [chol],
                "PEG": [peg],
                "TFR ": [tfr],
                "FRR": [frr],
                "BUFFER": [buffer],
                "OUTPUT": [1],
            })
            st.markdown("**Input data**")
            st.write(input_data)
            predictions = model.predict(input_data_)
            size, pdi = predictions[0]
            st.markdown("**Model predictions**")
            if size > 500 or pdi > 0.5:
                st.write("The sistem doesn't form")
                st.write(f"`OUTPUT`: 0")
            else:
                st.write(f"`SIZE`: {size:.2f}")
                st.write(f"`PDI`: {pdi:.2f}")
    
    # 2.3 Inverse model
    elif option == "Inverse model":
        st.header("Inverse model")
        st.markdown("This inference method pivots the pretrained XGBoost model to solve an [inverse problem](https://en.wikipedia.org/wiki/Inverse_problem): given target `SIZE` and `PDI` returns predictions for the other numerical features.")
        st.warning(":male-technologist: Work in progress... Only numerical predictions available.")
        with open(inverse_xgboost_model, "rb") as file:
            model = pickle.load(file)
        st.write(f"Loaded {inverse_xgboost_model} with the folloing performance metrics. The key metric is the square root of the error (either the MSE or the MAE).")
        st.table(pd.DataFrame({
            "Metric": ["R-squared", "Mean Squared Error", "Mean Absolute Error"],
            "Value": [0.11767681688070297, 89.26007080078125, 3.7459394931793213]
        }))
        if st.button("Predict"):
            input_data = pd.DataFrame({"SIZE": [size],"PDI": [pdi]})
            predictions = model.predict(input_data)
            predictions = np.abs(predictions)
            predictions = np.where(predictions < 0.5, 0, predictions)
            st.markdown("**Input data**")
            st.write(input_data)
            st.markdown("**Model predictions**")
            prediction_df = pd.DataFrame(predictions, columns=["ESM", "HSPC", "CHOL", "PEG", "TFR", "FRR"])
            st.write(prediction_df)
        
    # 2.4 Advanced models
    elif option == "Advanced models":
        st.header("Advanced models")
        st.write("Multiple models working in parallel for targeted predictions.") 
        st.subheader("Preview of some available advanced models for predicting `SIZE` or `PDI`")
        st.write("`ensemble-pdi`")
        st.table(pd.DataFrame({
            "Metric": ["R-squared", "Mean Squared Error (MSE)", "Mean Absolute Error (MAE)"],
            "Value": [0.5328156582541348, 0.00245118805677467875, 0.04254484741522172]
        }))
        st.write("`ensemble-size`")
        st.table(pd.DataFrame({
            "Metric": ["R-squared", "Mean Squared Error", "Mean Absolute Error"],
            "Value": [0.874431019712665, 340.2617544655023, 11.834716059736861]
        }))
        st.write("With an improvement in terms of metrics given by the following plot. The key metric is the square root of the error (either the MSE or the MAE).")
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
        st.subheader("Try the `ensemble-size` model")
        with open(size_model, "rb") as file:
            model = pickle.load(file)
        if st.button("Predict"):
            input_data = pd.DataFrame({
                "ML": [ml],
                "CHIP": [chip],
                "TLP": [tlp],
                "ESM": [0.0 if esm_disabled else esm],  
                "HSPC": [0.0 if hspc_disabled else hspc],  
                "CHOL": [chol],
                "PEG": [peg],
                "TFR ": [tfr],
                "FRR": [frr],
                "BUFFER": [buffer],
            })
            input_data_ = pd.DataFrame({
                "TLP": [tlp],
                "ESM": [esm],
                "HSPC": [hspc],
                "CHOL": [chol],
                "PEG": [peg],
                "TFR ": [tfr],
                "FRR": [frr],
            })
            st.markdown("**Input data**")
            st.write(input_data)
            st.markdown("**Model predictions**")
            if model.predict(input_data_) > 500:
                st.write("The sistem doesn't form")
                st.write(f"`OUTPUT`: 0")                
            else:
                st.write(f"Predicted `SIZE`: {model.predict(input_data_)}")   
        # 2.4.2 ensemble-pdi
        st.subheader("Try the `ensemble-pdi` model")
        with open(pdi_model, "rb") as file:
            model = pickle.load(file)
        if st.button("Predict", key='adv_pred_2'):
            input_data = pd.DataFrame({
                "ML": [ml],
                "CHIP": [chip],
                "TLP": [tlp],
                "ESM": [0.0 if esm_disabled else esm],  
                "HSPC": [0.0 if hspc_disabled else hspc],  
                "CHOL": [chol],
                "PEG": [peg],
                "TFR ": [tfr],
                "FRR": [frr],
                "BUFFER": [buffer],
            })
            input_data_ = pd.DataFrame({
                "TLP": [tlp],
                "ESM": [esm],
                "HSPC": [hspc],
                "CHOL": [chol],
                "PEG": [peg],
                "TFR ": [tfr],
                "FRR": [frr],
            })
            st.markdown("**Input data**")
            st.write(input_data)
            st.markdown("**Model predictions**")
            if model.predict(input_data_) > 0.5:
                st.write("The sistem doesn't form")
                st.write(f"`OUTPUT`: 0")                
            else:
                st.write(f"Predicted `PDI`: {model.predict(input_data_)}")


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
        #numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
        #df_numeric = data[numerical_cols]
        st.subheader("Numerical columns summary")
        st.write(numeric_data.describe())
        scaler = StandardScaler()
        df_standardized = pd.DataFrame(scaler.fit_transform(numeric_data), columns=numeric_data.columns)
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
        st.write("Displays the [flow between categorical data](https://en.wikipedia.org/wiki/Alluvial_diagram). Here the target variables `SIZE` and `PDI` are made categorical by grupping them into the following categories:")
        st.markdown("- **S** (Small) if `SIZE` is less than 100;")
        st.markdown("- **M** (Medium) if `SIZE` is between 100 and 199;") 
        st.markdown("- **L** (Large) if `SIZE` is 200 or more;")
        st.markdown("- **HMD** (Highly Monodisperse) if `PDI` is less than 0.1;")  
        st.markdown("- **MD** (Monodisperse) if `PDI` is between 0.1 and 0.25;")  
        st.markdown("- **PLD** (Polydisperse) if `PDI` is 0.25 or more.")  
        data_ = data.copy()
        for col in data_.columns:
            if data_[col].dtype == 'object':
                data_[col] = data_[col].astype(str)
                for col in data_.columns:
                    if data_[col].dtype == 'object':
                        data_[col] = data[col].astype(str)
        data_['SIZE'] = data_['SIZE'].apply(categorize_size)
        data_['PDI'] = data_['PDI'].apply(categorize_pdi)
        # 3.3.1 Sankey diagram  of two variables
        st.subheader("Sankey diagram of two variables")
        source = st.selectbox("Choose the first categorical variable of interest.", ("ML", "CHIP", "BUFFER", "OUTPUT", "SIZE", "PDI"))
        target = st.selectbox("Choose the second categorical variable of interest (different from the first).", ("CHIP", "ML", "BUFFER", "OUTPUT", "SIZE", "PDI"))
        if source == target:
            st.write(":red[INPUT ERROR. The selected variables cannot be equal.]")
        elif source != target:
            value_counts = data_.groupby([source, target]).size().reset_index(name='value')
            categories = list(set(value_counts[source]).union(set(value_counts[target])))
            category_to_index = {category: i for i, category in enumerate(categories)}
            sources = value_counts[source].map(category_to_index).tolist()
            targets = value_counts[target].map(category_to_index).tolist()
            values = value_counts['value'].tolist()
            sankey_fig = go.Figure(data=[go.Sankey(node=dict(pad=15,thickness=20,line=dict(color="black", width=0.5),label=categories),
                                                   link=dict(source=sources,target=targets,value=values))])
            st.plotly_chart(sankey_fig)
        # 3.3.2 Sankey diagram of more then two flows
        st.subheader("Sankey diagram of a customizable categorical flow")
        default_selection = ['ML', 'CHIP', 'BUFFER', 'OUTPUT']
        selected_columns = st.multiselect("Select categorical variables to visualize:", options=['ML', 'CHIP', 'BUFFER', 'OUTPUT', 'SIZE', 'PDI'], default=default_selection)
        if len(selected_columns) < 3:
            st.warning("Please select more then two two categorical variables.")
        else:
            value_counts = data_.groupby(selected_columns).size().reset_index(name='value')
            all_categories = []
            for col in selected_columns:
                all_categories.extend(value_counts[col].unique())
            all_categories = list(set(all_categories)) 
            category_to_index = {category: index for index, category in enumerate(all_categories)}
            sources = []
            targets = []
            values = []
            for index, row in value_counts.iterrows():
                for i in range(len(selected_columns) - 1):  
                    sources.append(category_to_index[row[selected_columns[i]]])
                    targets.append(category_to_index[row[selected_columns[i + 1]]])
                    values.append(row['value'])
            fig = go.Figure(data=[go.Sankey(node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5),label=all_categories),
                                            link=dict(source=sources, target=targets, value=values))])
            st.plotly_chart(fig, key="cust_sank")
    
    # 3.4 Feature importance
    elif option == "Feature importance":
        st.header("Feature importance with a Random Forest Regressor")
        st.markdown("Displays the [importance of each feature](https://en.wikipedia.org/wiki/Feature_selection) for predicting a set of input targets.")
        # 3.4.1 Single target feature importance
        st.subheader("Single target feature importance")
        #target_feature = st.selectbox("Select a target feature.", ("TLP", "ESM", "HSPC", "CHOL", "PEG", "FRR", "SIZE", "PDI"))
        target_feature = st.selectbox("Select a target feature.", ("SIZE", "PDI"))
        numeric_data_ = numeric_data.copy()
        if target_feature == "SIZE":
            numeric_data_ = numeric_data.drop(columns=["PDI"])
        if target_feature == "PDI":
            numeric_data_ = numeric_data.drop(columns=["SIZE"])        
        X = numeric_data_.drop(columns=[target_feature])
        if "SIZE" in X.columns:
            X = X.drop(columns=["SIZE"])
        if "PDI" in X.columns:
            X = X.drop(columns=["PDI"])
        y = numeric_data_[target_feature]  
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        feature_importances = pd.DataFrame({'Feature': X.columns,'Importance': rf.feature_importances_}).sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importances, x='Importance', y='Feature', hue='Feature', palette='viridis', dodge=False)
        st.pyplot(plt)
        # 3.4.2 Two targets feature importance
        target_feature_1 = st.selectbox("Select the first target feature.", ("SIZE", "PDI"))#("TLP", "ESM", "HSPC", "CHOL", "PEG", "FRR", "SIZE", "PDI")
        target_feature_2 = st.selectbox("Select the first target feature.", ("PDI", "SIZE"))#("TLP", "ESM", "HSPC", "CHOL", "PEG", "FRR", "SIZE", "PDI")
        if target_feature_1 == target_feature_2:
            st.error("INPUT ERROR: The selected variables cannot be the same.")
        elif target_feature_1 != target_feature_2:
            correlation_matrix = numeric_data.corr()
            filtered_data = numeric_data.copy()
            #if target_feature_1 == "SIZE" or target_feature_2 == "SIZE":
            #    filtered_data = filtered_data.drop(columns=["PDI"], errors='ignore')
            #if target_feature_1 == "PDI" or target_feature_2 == "PDI":
            #    filtered_data = filtered_data.drop(columns=["SIZE"], errors='ignore')
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
        st.markdown("""Displays the principal data features and their clusters using a standard [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) algorithm, a [UMAP](https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction#Uniform_manifold_approximation_and_projection), 
        and [k-means](https://en.wikipedia.org/wiki/K-means_clustering) clustering. These plots help in model design, pinpoint processing requirements and can provide guidance during training and validation.""")
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
        n_clusters = st.slider("Select number of clusters for K-Means:", min_value=2, max_value=10, value=3)
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
