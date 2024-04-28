import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import plotly.express as px
import plotly.graph_objects as go
from category_encoders import TargetEncoder


# Set page config
st.set_page_config(page_title='Regression Analysis System', layout='wide')

# Title and description
st.title('Regression Analysis for Price Prediction')
st.markdown('''
This application predicts prices using various regression models. Upload your dataset or use the preloaded data for demonstration.
''')

# Load and preprocess data
@st.cache
def load_data(filename):
    return pd.read_csv(filename).select_dtypes(exclude=['datetime', 'datetimetz'])  # Exclude date columns

uploaded_file = st.sidebar.file_uploader("Choose a file, or leave blank to use default data")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = load_data('data/merged_leads_land_not_null.csv')

# Sidebar - Model setup
st.sidebar.header("Model Setup")
target = st.sidebar.selectbox('Select the target variable for prediction:', df.columns, index=df.columns.get_loc('price_usd') if 'price_usd' in df.columns else 0)
features = st.sidebar.multiselect('Select features to use:', df.columns.drop(target), default=df.columns.drop(target).tolist())
encode_categorical = st.sidebar.radio("Encode categorical variables:", ["Frequency Encoding", "Target encoding"], index=0)
model_type = st.sidebar.selectbox('Choose a regression model:', [
    'Linear Regression', 'Ridge', 'Lasso', 'Random Forest Regressor',
    'Decision Tree Regressor', 'XGBoost Regressor', 'AdaBoost Regressor',
    'Gradient Boosting Regressor'
])

# Preprocessing configuration
numeric_cols = make_column_selector(dtype_include=np.number)
categorical_cols = make_column_selector(dtype_include=object)
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
categorical_transformer = TargetEncoder() if encode_categorical == "TargetEncoder" else Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('freq_enc', FunctionTransformer(lambda X: pd.DataFrame(X).apply(lambda x: x.map(x.value_counts(normalize=True)), axis=0).values, validate=False))])
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
], remainder='passthrough')

# Train-test split and model training
if target in df.columns and features:
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
    model = model_dict[model_type]
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Model performance metrics
    st.header('Model Performance')
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f'Mean Squared Error (MSE): {mse}')
    st.write(f'Mean Absolute Error (MAE): {mae}')
    st.write(f'R^2 Score: {r2:.4f}')

    # Visualization of Actual vs Predicted
    fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'}, title='Actual vs Predicted')
    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', line=dict(color='red', dash='dash'), name='Ideal Fit'))
    st.plotly_chart(fig)

    # Feature importance or model coefficients visualization
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = model.coef_
    else:
        importance = [0]*len(features)  # Default to zeros if no importance or coefficients

    features = X_train.columns
    fig_importance = px.bar(x=features, y=importance, labels={'x': 'Features', 'y': 'Importance or Weight'}, title='Feature Importance or Weights')
    st.plotly_chart(fig_importance)

    # Feature importance table with download option
    feature_importance_df = pd.DataFrame({
        'Features': features,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    st.write("Feature Importances")
    st.dataframe(feature_importance_df)
    csv = feature_importance_df.to_csv().encode('utf-8')
    st.download_button(label="Download Feature Importances", data=csv, file_name='feature_importances.csv', mime='text/csv')
else:
    st.error("Please ensure the dataset includes the target column and select at least one feature.")
