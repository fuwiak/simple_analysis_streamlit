import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title='Regression Analysis System', layout='wide')

# Title and description
st.title('Regression Analysis for Price Prediction')
st.markdown('''
This application allows you to select different regression models to predict the `price_usd` based on available data.
Please upload your dataset or use the default loaded data.
''')

# Load data
@st.cache
def load_data():
    data = pd.read_csv('data/your_data.csv')
    return data

df = load_data()

# Sidebar for model selection
st.sidebar.subheader('Model Selection')
model_type = st.sidebar.selectbox('Choose a regression model:', [
    'Linear Regression', 'Ridge', 'Lasso', 'Random Forest Regressor',
    'Decision Tree Regressor', 'XGBoost Regressor', 'AdaBoost Regressor',
    'Gradient Boosting Regressor'
])

# Preprocessing
st.subheader('Data Preprocessing')
if 'price_usd' in df.columns:
    X = df.drop('price_usd', axis=1)
    y = df['price_usd']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
else:
    st.error('The selected column `price_usd` is not in the dataset.')

# Regression model mapping
model_dict = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'Random Forest Regressor': RandomForestRegressor(),
    'Decision Tree Regressor': DecisionTreeRegressor(),
    'XGBoost Regressor': XGBRegressor(),
    'AdaBoost Regressor': AdaBoostRegressor(),
    'Gradient Boosting Regressor': GradientBoostingRegressor()
}

# Model training and prediction
if model_type in model_dict:
    model = model_dict[model_type]
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader(f'Results using {model_type}')
    st.write(f'Mean Squared Error: {mse:.2f}')
    st.write(f'Mean Absolute Error: {mae:.2f}')
    st.write(f'R^2 Score: {r2:.2f}')

    # Plot results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predicted vs Actual'))
    fig.update_layout(title='Prediction Accuracy', xaxis_title='Actual', yaxis_title='Predicted')
    st.plotly_chart(fig)

    # Feature Importance (if applicable)
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        labels = X.columns[indices]
        fig_importance = px.bar(x=labels, y=importance[indices], labels={'x': 'Features', 'y': 'Importance'},
                                title='Feature Importance')
        st.plotly_chart(fig_importance)
