import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import plotly.express as px
import scipy.cluster.hierarchy as sch
from plotly import graph_objects as go

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
    # 2.1 Sankey diagram  of two variables
    st.subheader("Sankey diagram of two variables")
    source = st.text_input("Choose the first categorical variable of interest. Answer ML, CHIP, BUFFER or OUTPUT:", "ML")
    target = st.text_input("Choose the second categorical variable of interest (different from the first). Answer ML, CHIP, BUFFER or OUTPUT:", "BUFFER")
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

elif option == "Pair Plot":
    st.header("Pair Plot")
    st.write("Displays pairwise relationships between numerical features.")

    # Plot pairplot
    plt.figure()
    sns.pairplot(data.select_dtypes(include=['float64', 'int64']))
    st.pyplot(plt)
