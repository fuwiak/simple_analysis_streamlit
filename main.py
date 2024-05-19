import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# Set page configuration
st.set_page_config(page_title='Logistics ML Analysis System', layout='wide')

# Title
st.title("Machine Learning for Logistics Management: Pricing, Budget Planning, and Demand Forecasting")

# Sidebar for file selection
st.sidebar.header("File Selection")
file_paths = {
    'Air Rates': 'data/rates_air.csv',
    'Land Rates': 'data/rates_land.csv',
    'Sea Rates': 'data/rates_sea.csv',
    'Merged Leads (default)': 'data/merged_leads_land_not_null.csv'
}
selected_file = st.sidebar.selectbox("Choose a dataset:", options=list(file_paths.keys()), index=3)
df = pd.read_csv(file_paths[selected_file])
st.sidebar.text(f"Loaded File: {selected_file}")

# Convert hs_code to string type if it exists in the dataframe
if 'hs_code' in df.columns:
    df['hs_code'] = df['hs_code'].astype(str)

# Display DataFrame with dynamic view control
num_rows = st.sidebar.slider('Number of rows to display:', min_value=5, max_value=100, value=20)
st.markdown("## Data Overview")
st.dataframe(df.head(num_rows), width=1500, height=600)

# Setup columns for stats and interactive plots
col1, col2, col3 = st.columns(3)

# Display basic data stats in a wider format
with col1:
    st.subheader("Data Summary")
    st.write("Number of rows:", df.shape[0])
    st.write("Number of columns:", df.shape[1])
    st.write("Missing values:", df.isnull().sum().sum())
    st.write("Duplicates:", df.duplicated().sum())

# Interactive plot setup with dynamic column selection
with col2:
    st.subheader("Interactive Data Plot")
    all_columns = df.columns.tolist()
    num_columns = df.select_dtypes(include=np.number).columns.tolist()
    cat_columns = df.select_dtypes(include='object').columns.tolist()

    x_axis = st.selectbox('Choose the X-axis:', all_columns, index=0)
    if x_axis in num_columns:
        y_axis = st.selectbox('Choose the Y-axis:', num_columns, index=1 if len(num_columns) > 1 else 0)
    else:
        y_axis = st.selectbox('Choose the Y-axis:', cat_columns, index=0)

    plot_type = st.radio("Select plot type:", ('Scatter Plot', 'Line Plot', 'Bar Plot'))
    if plot_type == 'Scatter Plot' and x_axis in num_columns and y_axis in num_columns:
        fig = px.scatter(df, x=x_axis, y=y_axis, title=f'Scatter Plot of {x_axis} vs {y_axis}')
    elif plot_type == 'Line Plot' and x_axis in num_columns and y_axis in num_columns:
        fig = px.line(df, x=x_axis, y=y_axis, title=f'Line Plot of {x_axis} vs {y_axis}')
    elif plot_type == 'Bar Plot' and x_axis in cat_columns:
        fig = px.bar(df, x=x_axis, title=f'Count Plot of {x_axis}')
    else:
        st.warning('Please select appropriate axes for the chosen plot type.')
        fig = None
    if fig:
        st.plotly_chart(fig)

# Histogram for any numerical column
with col3:
    st.subheader("Data Distribution Plot")
    selected_column = st.selectbox('Select a numerical column for histogram:', num_columns)
    if selected_column:
        hist_fig = px.histogram(df, x=selected_column, title=f'Histogram of {selected_column}')
        st.plotly_chart(hist_fig)

# Pie chart for categorical data
if cat_columns:
    st.markdown("## Categorical Data Composition")
    categorical_column = st.selectbox('Select a categorical column:', cat_columns, index=0)
    if df[categorical_column].nunique() < 10:  # Only make a pie chart if there are fewer than 10 unique categories
        pie_fig = px.pie(df, names=categorical_column, title=f'Pie Chart of {categorical_column}')
        st.plotly_chart(pie_fig)
    else:
        st.write("Selected column has too many unique categories for a pie chart.")

# Data Report
st.markdown("## Data Report")
st.write("Descriptive Statistics:")
st.dataframe(df.describe())

# Correlation matrix for numerical data
show_corr = st.checkbox('Show correlation matrix for numerical data', value=True)
if show_corr:
    st.subheader("Correlation Matrix")
    numerical_data = df.select_dtypes(include=np.number)
    if not numerical_data.empty:
        corr_matrix = numerical_data.corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect='auto', title='Correlation Matrix of Numerical Features')
        st.plotly_chart(fig_corr)
    else:
        st.write("No numerical data available for correlation.")

# Downloadable data report
st.markdown("## Download Data Report")

@st.cache_data
def convert_df_to_csv(d):
    return d.to_csv().encode('utf-8')
csv = convert_df_to_csv(df)
st.download_button("Download Data Report", csv, "data_report.csv", "text/csv", key='download-csv')
