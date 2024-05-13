# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.decomposition import PCA, TruncatedSVD
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# import plotly.graph_objects as go
#
# st.set_page_config(page_title='Logistics ML Analysis System', layout='wide')
#
# st.title('Clustering Analysis')
#
# # Load and preprocess data
# df = pd.read_csv('data/merged_leads_land_not_null.csv')
# numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
# df_numeric = df[numeric_columns].dropna()
# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(df_numeric)
#
# # Dimensionality reduction choice
# col1, col2 = st.columns([1, 2])
# with col1:
#     st.subheader('Dimensionality Reduction')
#     reduction_method = st.radio('Select dimensionality reduction method:', ('PCA', 'SVD'))
#
# if reduction_method == 'PCA':
#     reducer = PCA(n_components=2)
# else:
#     reducer = TruncatedSVD(n_components=2)
#
# data_reduced = reducer.fit_transform(data_scaled)
# st.write('Processed Data:', pd.DataFrame(data_reduced, columns=['Component 1', 'Component 2']))
#
# # Determine the number of clusters using the elbow method
# k_values = range(1, 11)
# inertias = []
# for k in k_values:
#     kmeans = KMeans(n_clusters=k, random_state=0)
#     kmeans.fit(data_reduced)
#     inertias.append(kmeans.inertia_)
#
# # Elbow plot configuration
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=list(k_values), y=inertias, mode='lines+markers', name='Inertia'))
# fig.add_vline(x=3, line_width=2, line_dash="dash", line_color="green")
# fig.update_layout(title='Elbow Method For Optimal k', xaxis_title='Number of Clusters', yaxis_title='Inertia')
#
# with col2:
#     st.subheader('Determine Optimal Number of Clusters')
#     st.plotly_chart(fig)
#
# # User chooses number of clusters
# n_clusters = st.slider('Select number of clusters:', min_value=2, max_value=10, value=3, key='cluster_slider')
#
# # Apply K-Means and visualize clusters
# kmeans = KMeans(n_clusters=n_clusters, random_state=0)
# clusters = kmeans.fit_predict(data_reduced)
# df_numeric['Cluster'] = clusters
#
# # Visualization of Clusters
# st.subheader('Cluster Visualization')
# fig = px.scatter(x=data_reduced[:, 0], y=data_reduced[:, 1], color=clusters.astype(str),
#                  labels={'x': 'Component 1', 'y': 'Component 2'}, title='Cluster Visualization')
# st.plotly_chart(fig)
#
# # Displaying statistics for each cluster
# cluster_stats = df_numeric.groupby('Cluster').describe().stack(level=0)
#
# # Description of what highlight colors mean
# st.markdown('### Статистика по кластерам')
# st.markdown('**Желтый цвет** выделяет максимальные значения, **светло-зеленый** - минимальные значения в каждом столбце.')
#
# # Improve the display of cluster statistics with formatting
# def highlight_max(s):
#     return ['background-color: yellow' if v == s.max() else '' for v in s]
#
# def highlight_min(s):
#     return ['background-color: lightgreen' if v == s.min() else '' for v in s]
#
# styled_stats = cluster_stats.style.apply(highlight_max).apply(highlight_min)
# st.dataframe(styled_stats)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title='Logistics ML Analysis System', layout='wide')

st.title('Clustering Analysis')

# Sidebar for user inputs
st.sidebar.header('Settings')
n_components = st.sidebar.selectbox('Select number of components:', options=[2, 3, 4], index=0)

# Load and preprocess data
df = pd.read_csv('data/merged_leads_land_not_null.csv')
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
df_numeric = df[numeric_columns].dropna()
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_numeric)

# Dimensionality reduction
reduction_method = st.sidebar.radio('Select dimensionality reduction method:', ('PCA', 'SVD'), key='reduction_method')

if reduction_method == 'PCA':
    reducer = PCA(n_components=n_components)
else:
    reducer = TruncatedSVD(n_components=n_components)

data_reduced = reducer.fit_transform(data_scaled)
st.write('Processed Data:', pd.DataFrame(data_reduced, columns=[f'Component {i+1}' for i in range(n_components)]))

# Clustering
k_values = range(1, 11)
inertias = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data_reduced)
    inertias.append(kmeans.inertia_)

# Elbow plot for determining k
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(k_values), y=inertias, mode='lines+markers', name='Inertia'))
fig.add_vline(x=3, line_width=2, line_dash="dash", line_color="green")
fig.update_layout(title='Elbow Method For Optimal k', xaxis_title='Number of Clusters', yaxis_title='Inertia')
st.plotly_chart(fig)

# Select number of clusters
n_clusters = st.slider('Select number of clusters:', min_value=2, max_value=10, value=3, key='cluster_slider')

# Apply K-Means and visualize clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
clusters = kmeans.fit_predict(data_reduced)
df_numeric['Cluster'] = clusters

# Visualization of Clusters
st.subheader('Cluster Visualization')
if n_components == 2:
    fig = px.scatter(x=data_reduced[:, 0], y=data_reduced[:, 1], color=clusters.astype(str),
                     labels={'x': 'Component 1', 'y': 'Component 2'}, title='Cluster Visualization')
elif n_components > 2:
    fig = px.scatter_3d(x=data_reduced[:, 0], y=data_reduced[:, 1], z=data_reduced[:, 2] if n_components > 2 else 0,
                        color=clusters.astype(str), labels={'x': 'Component 1', 'y': 'Component 2', 'z': 'Component 3' if n_components > 2 else 'Component 2'},
                        title='Cluster Visualization')
st.plotly_chart(fig)

# Displaying statistics for each cluster
cluster_stats = df_numeric.groupby('Cluster').describe().stack(level=0)

# Description of what highlight colors mean
st.markdown('### Статистика по кластерам')
st.markdown('**Желтый цвет** выделяет максимальные значения, **светло-зеленый** - минимальные значения в каждом столбце.')

# Improve the display of cluster statistics with formatting
def highlight_max(s):
    return ['background-color: yellow' if v == s.max() else '' for v in s]

def highlight_min(s):
    return ['background-color: lightgreen' if v == s.min() else '' for v in s]

styled_stats = cluster_stats.style.apply(highlight_max).apply(highlight_min)
st.dataframe(styled_stats)
