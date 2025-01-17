import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import plotly.express as px
import scipy.cluster.hierarchy as sch

# Load data
file_path = 'data.csv'
data = pd.read_csv(file_path, encoding='latin1')
data = data.set_index('ID')

# Streamlit app title
st.title("Data Visualization Dashboard")
st.write("Data explorations with smart visualizations, including correlation heatmaps, alluvional plots, and more.")

# Sidebar for navigation
st.sidebar.title("Visualization Options")
option = st.sidebar.selectbox(
    "Choose a Visualization:",
    [
        "Correlation heatmaps",
        "Alluvial Plot",
        
    ],
)

# Visualizations
# 1. Correlation heatmap
if option == "Correlation heatmaps":
    st.header("Correlation heatmap")
    st.write("Displays the correlation between numerical features in the dataset.")
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

elif option == "Alluvial Plot":
    st.header("Alluvial Plot")
    st.write("Displays the flow of categorical data using an alluvial plot.")
    formed = st.text_input("Choose the first categorical variable of interest. Answer ML, CHIP, BUFFER or OUTPUT:", "ML")
    # Example columns
    col1 = st.selectbox("Select the first categorical column:", data.select_dtypes(include=['object']).columns)
    col2 = st.selectbox("Select the second categorical column:", data.select_dtypes(include=['object']).columns)
    col3 = st.selectbox("Select the third categorical column (optional):", ["None"] + data.select_dtypes(include=['object']).columns.tolist())

    # Plot alluvial diagram
    alluvial_data = data[[col1, col2] + ([col3] if col3 != "None" else [])].dropna()
    fig = px.parallel_categories(alluvial_data, dimensions=[col1, col2] + ([col3] if col3 != "None" else []))
    st.plotly_chart(fig)

elif option == "Pair Plot":
    st.header("Pair Plot")
    st.write("Displays pairwise relationships between numerical features.")

    # Plot pairplot
    plt.figure()
    sns.pairplot(data.select_dtypes(include=['float64', 'int64']))
    st.pyplot(plt)
