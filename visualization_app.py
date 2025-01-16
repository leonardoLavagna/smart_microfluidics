import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import plotly.express as px

file_path = 'data.csv'
data = pd.read_csv(file_path, encoding='latin1')
data = data.set_index('ID')

# Streamlit app title
st.title("Data Visualization Dashboard")
st.write("Explore visualizations of your data, including treemaps, correlation heatmaps, and more.")

# Sidebar for navigation
st.sidebar.title("Visualization Options")
option = st.sidebar.selectbox(
    "Choose a Visualization:",
    [
        "Correlation Heatmap",
        "Tree Map",
        "Alluvial Plot",
        "Pair Plot",
    ],
)

# Define visualizations
if option == "Correlation Heatmap":
    st.header("Correlation Heatmap")
    st.write("Displays the correlation between numerical features in the dataset.")

    # Compute correlation
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numeric_data.corr()

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    st.pyplot(plt)

elif option == "Tree Map":
    st.header("Tree Map")
    st.write("Displays hierarchical data visualization using a treemap.")

    # Define categories and size columns
    category_col = st.selectbox("Select a categorical column:", data.select_dtypes(include=['object']).columns)
    size_col = st.selectbox("Select a size column:", data.select_dtypes(include=['float64', 'int64']).columns)

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
