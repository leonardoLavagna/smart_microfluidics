import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import plotly.express as px

# Load data
file_path = 'data.csv'
data = pd.read_csv(file_path, encoding='latin1')
data = data.set_index('ID')

# Streamlit app title
st.title("Data Visualization Dashboard")
st.write("Data explorations with smart visualizations, including treemaps, correlation heatmaps, and more.")

# Sidebar for navigation
st.sidebar.title("Visualization Options")
option = st.sidebar.selectbox(
    "Choose a Visualization:",
    [
        "Correlation heatmap",
        "Clustered correlation heatmap"
        
    ],
)

# Visualizations
# 1. Correlation heatmap
if option == "Correlation heatmap":
    st.header("Correlation heatmap")
    st.write("Displays the correlation between numerical features in the dataset.")
    formed = st.text_input("Are you interested in the dataset of formed liposomes? Answer YES (Y) or NO (N):", "YES")
    n_ids = st.number_input("Enter the number of formulations you desire to visualize:", min_value=2, max_value=20, value=10)
    st.subheader("Correlations by features")
    if formed == "YES" or formed == "Y":
        # Select numeric columns for correlation calculation corresponding to formation
        numeric_cols = data.select_dtypes(include=['number']).columns
        yes_data = data[data['OUTPUT'] == 'YES']
        numeric_yes_data = yes_data[numeric_cols]
        # Plot heatmap
        correlation_matrix = numeric_yes_data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        st.pyplot(plt)
        # Plot heatmap by IDs
        st.subheader("Correlations by IDs")
        correlation_matrix_t = numeric_yes_data.T.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        st.pyplot(plt)
    else:
        # Select numeric columns for correlation calculation corresponding to no-formation
        numeric_cols = data.select_dtypes(include=['number']).columns
        yes_data = data[data['OUTPUT'] == 'NO']
        numeric_no_data = no_data[numeric_cols]
        # Plot heatmap
        correlation_matrix = numeric_no_data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        st.pyplot(plt)        
        # Plot heatmap by IDs
        st.subheader("Correlations by IDs")
        correlation_matrix_t = numeric_no_data.T.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix_t, annot=True, fmt=".2f", cmap="coolwarm")
        st.pyplot(plt)

# 2. Clustered correlation heatmap
elif option == "Clustered correlation heatmap":
    st.header("Clustered correlation heatmapp")
    st.write("Displays hierarchies in the correlation heatmap via dendrograms.")
    num_ids = st.number_input("Enter the number of formulations you desire to visualize:", min_value=2, max_value=20, value=10)
    formed = st.text_input("Are you interested in the dataset of formed liposomes? Answer YES (Y) or NO (N):", "YES")
    if formed == "YES" or formed == "Y":
        # Select numeric columns for correlation calculation corresponding to formation
        data_red = data[:10]
        numeric_cols = data_red.select_dtypes(include=['number']).columns
        yes_data = data_red[data_red['OUTPUT'] == 'YES']
        numeric_yes_data_red = yes_data[numeric_cols]

        # Prepare treemap data
    treemap_data = data.groupby(category_col)[size_col].sum().reset_index()

    # Plot treemap
    fig = px.treemap(
        treemap_data,
        path=[category_col],
        values=size_col,
        title=f"Treemap of {category_col} by {size_col}",
    )
    st.plotly_chart(fig)

elif option == "Alluvial Plot":
    st.header("Alluvial Plot")
    st.write("Displays the flow of categorical data using an alluvial plot.")

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
