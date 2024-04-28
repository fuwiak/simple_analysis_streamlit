import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# Set page configuration
st.set_page_config(page_title='Logistics ML Analysis System', layout='wide')

# Title
st.title(
    "Machine Learning for Logistics Management: Pricing, Budget Planning, and Demand Forecasting")

# Load data
path = 'merged_leads_land_not_null.csv'
df = pd.read_csv(f"data/{path}")

# Display DataFrame with wider view
st.markdown("## Data Overview")
st.dataframe(df, width=1500, height=600)

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
    x_axis = st.selectbox('Choose the X-axis:', all_columns, index=0)
    y_axis = st.selectbox('Choose the Y-axis:', all_columns,
                          index=1 if len(all_columns) > 1 else 0)
    plot_type = st.radio("Select plot type:",
                         ('Scatter Plot', 'Line Plot', 'Bar Plot'))

    if plot_type == 'Scatter Plot':
        fig = px.scatter(df, x=x_axis, y=y_axis,
                         title=f'Scatter Plot of {x_axis} vs {y_axis}')
    elif plot_type == 'Line Plot':
        fig = px.line(df, x=x_axis, y=y_axis,
                      title=f'Line Plot of {x_axis} vs {y_axis}')
    else:
        fig = px.bar(df, x=x_axis, y=y_axis,
                     title=f'Bar Plot of {x_axis} vs {y_axis}')

    st.plotly_chart(fig)

# Histogram for any numerical column
with col3:
    st.subheader("Data Distribution Plot")
    # Set 'price_usd' as default if it exists
    num_columns = df.select_dtypes(include=np.number).columns
    default_num_col = 'price_usd' if 'price_usd' in num_columns else \
    num_columns[0]
    selected_column = st.selectbox('Select a column for histogram:',
                                   num_columns,
                                   index=num_columns.get_loc(default_num_col))
    hist_fig = px.histogram(df, x=selected_column,
                            title=f'Histogram of {selected_column}')
    st.plotly_chart(hist_fig)

# Pie chart for categorical data
if df.select_dtypes(include='object').columns.any():
    st.markdown("## Categorical Data Composition")
    # Set 'container_type' as default if it exists
    cat_columns = df.select_dtypes(include='object').columns
    default_cat_col = 'container_type' if 'container_type' in cat_columns else \
    cat_columns[0]
    categorical_column = st.selectbox('Select a categorical column:',
                                      cat_columns, index=cat_columns.get_loc(
            default_cat_col))
    if df[
        categorical_column].nunique() < 10:  # Only make a pie chart if there are fewer than 10 unique categories
        pie_fig = px.pie(df, names=categorical_column,
                         title=f'Pie Chart of {categorical_column}')
        st.plotly_chart(pie_fig)
    else:
        st.write(
            "Selected column has too many unique categories for a pie chart.")

# Data Report
st.markdown("## Data Report")
st.write("Descriptive Statistics:")
st.dataframe(df.describe())

# Correlation matrix for numerical data
show_corr = st.checkbox('Show correlation matrix for numerical data',
                        value=True)
if show_corr:
    st.subheader("Correlation Matrix")
    numerical_data = df.select_dtypes(include=np.number)
    if not numerical_data.empty:
        corr_matrix = numerical_data.corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect='auto',
                             title='Correlation Matrix of Numerical Features')
        st.plotly_chart(fig_corr)
    else:
        st.write("No numerical data available for correlation.")

# Downloadable data report
st.markdown("## Download Data Report")


@st.cache
def convert_df_to_csv(d):
    return d.to_csv().encode('utf-8')


csv = convert_df_to_csv(df)
st.download_button("Download Data Report", csv, "data_report.csv", "text/csv",
                   key='download-csv')
