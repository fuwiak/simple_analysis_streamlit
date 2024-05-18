import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, \
    RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, \
    AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import plotly.express as px
import plotly.graph_objects as go
from category_encoders import TargetEncoder
# import smogn
from time import time
from sklearn.utils import resample
import logging

# Set page config
st.set_page_config(page_title='Regression Analysis System', layout='wide')

# Title and description
st.title('Regression Analysis for Price Prediction')
st.markdown('''
This application predicts prices using various regression models. Upload your dataset or use the preloaded data for demonstration.
''')


# Load and preprocess data
@st.cache_data
def load_data(filename):
    return pd.read_csv(filename)


uploaded_file = st.sidebar.file_uploader(
    "Choose a file, or leave blank to use default data")

# Initialize origin and destination columns
origin_col = 'port_from'
destination_col = 'port_to'

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'rate_sea.csv' in uploaded_file.name:
        origin_col = 'origin_country'
        destination_col = 'destination_country'
else:
    df = load_data('data/leads.csv')
    drop_columns = ['request_date']
    df.drop(columns=drop_columns, inplace=True, errors='ignore')

# Display available connections
st.sidebar.header("Available Connections")
if origin_col in df.columns and destination_col in df.columns:
    available_connections = df[[origin_col, destination_col]].drop_duplicates()
    st.sidebar.dataframe(available_connections, width=300, height=200)
else:
    st.sidebar.warning(
        f"Columns {origin_col} and/or {destination_col} not found in the dataset.")

# Exclude datetime columns
datetime_columns = df.select_dtypes(
    include=['datetime', 'datetimetz']).columns.tolist()

# Select only numeric columns
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = df.select_dtypes(include=[object]).columns.tolist()

# Sidebar - Model setup
st.sidebar.header("Model Setup")
target = st.sidebar.selectbox('Select the target variable for prediction:',
                              df.columns, index=df.columns.get_loc(
        'price_usd') if 'price_usd' in df.columns else 0)
features = st.sidebar.multiselect('Select features to use:',
                                  df.columns.drop([target] + datetime_columns),
                                  default=df.columns.drop(
                                      [target] + datetime_columns).tolist())
encode_categorical = st.sidebar.radio("Encode categorical variables:",
                                      ["Frequency Encoding",
                                       "Target encoding"], index=0)

model_type = st.sidebar.selectbox('Choose a regression model:', [
    'Ridge', 'Lasso', 'Random Forest Regressor',
    'Decision Tree Regressor', 'XGBoost Regressor', 'AdaBoost Regressor',
    'Gradient Boosting Regressor'
])

# Hyperparameters Sidebar
alpha_ridge, solver_ridge, max_iter_ridge, alpha_lasso, max_iter_lasso, n_estimators_rf, max_depth_rf, min_samples_split_rf, min_samples_leaf_rf, max_depth_dt, min_samples_split_dt, min_samples_leaf_dt, learning_rate_xgb, n_estimators_xgb, max_depth_xgb, subsample_xgb, colsample_bytree_xgb, n_estimators_ada, learning_rate_ada, loss_ada, n_estimators_gb, learning_rate_gb, max_depth_gb, min_samples_split_gb, min_samples_leaf_gb = [
                                                                                                                                                                                                                                                                                                                                                                                                                                                    None] * 25

if model_type == 'Ridge':
    alpha_ridge = st.sidebar.slider('Alpha', 0.01, 10.0, 1.0)
    solver_ridge = st.sidebar.selectbox('Solver',
                                        ['auto', 'svd', 'cholesky', 'lsqr',
                                         'sparse_cg', 'sag', 'saga'])
    max_iter_ridge = st.sidebar.slider('Max Iterations', 100, 10000, 1000)
elif model_type == 'Lasso':
    alpha_lasso = st.sidebar.slider('Alpha', 0.01, 10.0, 1.0)
    max_iter_lasso = st.sidebar.slider('Max Iterations', 100, 10000, 1000)
elif model_type == 'Random Forest Regressor':
    n_estimators_rf = st.sidebar.slider('Number of Trees', 100, 500, 100)
    max_depth_rf = st.sidebar.slider('Max Depth', 1, 20, 10)
    min_samples_split_rf = st.sidebar.slider('Min Samples Split', 2, 10, 2)
    min_samples_leaf_rf = st.sidebar.slider('Min Samples Leaf', 1, 5, 1)
elif model_type == 'Decision Tree Regressor':
    max_depth_dt = st.sidebar.slider('Max Depth', 1, 20, 10)
    min_samples_split_dt = st.sidebar.slider('Min Samples Split', 2, 10, 2)
    min_samples_leaf_dt = st.sidebar.slider('Min Samples Leaf', 1, 5, 1)
elif model_type == 'XGBoost Regressor':
    learning_rate_xgb = st.sidebar.slider('Learning Rate', 0.01, 0.5, 0.1)
    n_estimators_xgb = st.sidebar.slider('Number of Estimators', 100, 500, 100)
    max_depth_xgb = st.sidebar.slider('Max Depth', 1, 10, 3)
    subsample_xgb = st.sidebar.slider('Subsample', 0.5, 1.0, 0.8)
    colsample_bytree_xgb = st.sidebar.slider('Colsample by Tree', 0.5, 1.0,
                                             0.8)
elif model_type == 'AdaBoost Regressor':
    n_estimators_ada = st.sidebar.slider('Number of Estimators', 50, 200, 50)
    learning_rate_ada = st.sidebar.slider('Learning Rate', 0.01, 1.0, 0.1)
    loss_ada = st.sidebar.selectbox('Loss Function',
                                    ['linear', 'square', 'exponential'])
elif model_type == 'Gradient Boosting Regressor':
    n_estimators_gb = st.sidebar.slider('Number of Estimators', 100, 500, 100)
    learning_rate_gb = st.sidebar.slider('Learning Rate', 0.01, 0.5, 0.1)
    max_depth_gb = st.sidebar.slider('Max Depth', 1, 10, 3)
    min_samples_split_gb = st.sidebar.slider('Min Samples Split', 2, 10, 2)
    min_samples_leaf_gb = st.sidebar.slider('Min Samples Leaf', 1, 5, 1)

# Hyperparameter tuning
st.sidebar.header("Hyperparameter Tuning")
search_type = st.sidebar.radio("Search Type",
                               ["None", "RandomizedSearch", "GridSearch"],
                               index=0)
n_iter_search = st.sidebar.number_input(
    "Number of iterations (for RandomizedSearch)", min_value=1, value=10)
cv_folds = st.sidebar.number_input("Number of CV folds", min_value=2, value=3)

# Filtering based on origin and destination
origin = st.sidebar.selectbox(f'Select {origin_col.replace("_", " ")}:',
                              df[origin_col].unique())
destination = st.sidebar.selectbox(
    f'Select {destination_col.replace("_", " ")}:',
    df[destination_col].unique())
filtered_df = df[
    (df[origin_col] == origin) & (df[destination_col] == destination)]

# Check if filtered_df is empty
if filtered_df.empty:
    st.error(
        "No data available for the selected origin and destination. Please select different options.")
else:
    # Preprocessing configuration
    numeric_cols = make_column_selector(dtype_include=np.number)
    categorical_cols = make_column_selector(dtype_include=object)
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    categorical_transformer = TargetEncoder() if encode_categorical == "Target encoding" else Pipeline(
        [
            ('imputer',
             SimpleImputer(strategy='constant', fill_value='missing')),
            ('freq_enc', FunctionTransformer(lambda X: pd.DataFrame(X).apply(
                lambda x: x.map(x.value_counts(normalize=True)),
                axis=0).values, validate=False))])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='passthrough')

    # Model Initialization with Hyperparameters
    model_dict = {
        'Ridge': Ridge(alpha=alpha_ridge, solver=solver_ridge,
                       max_iter=max_iter_ridge) if model_type == 'Ridge' else Ridge(),
        'Lasso': Lasso(alpha=alpha_lasso,
                       max_iter=max_iter_lasso) if model_type == 'Lasso' else Lasso(),
        'Random Forest Regressor': RandomForestRegressor(
            n_estimators=n_estimators_rf, max_depth=max_depth_rf,
            min_samples_split=min_samples_split_rf,
            min_samples_leaf=min_samples_leaf_rf) if model_type == 'Random Forest Regressor' else RandomForestRegressor(),
        'Decision Tree Regressor': DecisionTreeRegressor(
            max_depth=max_depth_dt, min_samples_split=min_samples_split_dt,
            min_samples_leaf=min_samples_leaf_dt) if model_type == 'Decision Tree Regressor' else DecisionTreeRegressor(),
        'XGBoost Regressor': XGBRegressor(learning_rate=learning_rate_xgb,
                                          n_estimators=n_estimators_xgb,
                                          max_depth=max_depth_xgb,
                                          subsample=subsample_xgb,
                                          colsample_bytree=colsample_bytree_xgb) if model_type == 'XGBoost Regressor' else XGBRegressor(),
        'AdaBoost Regressor': AdaBoostRegressor(n_estimators=n_estimators_ada,
                                                learning_rate=learning_rate_ada,
                                                loss=loss_ada) if model_type == 'AdaBoost Regressor' else AdaBoostRegressor(),
        'Gradient Boosting Regressor': GradientBoostingRegressor(
            n_estimators=n_estimators_gb, learning_rate=learning_rate_gb,
            max_depth=max_depth_gb, min_samples_split=min_samples_split_gb,
            min_samples_leaf=min_samples_leaf_gb) if model_type == 'Gradient Boosting Regressor' else GradientBoostingRegressor()
    }

    param_distributions = {
        'Ridge': {'model__alpha': np.linspace(0.01, 10.0, 100),
                  'model__solver': ['auto', 'svd', 'cholesky', 'lsqr',
                                    'sparse_cg', 'sag', 'saga']},
        'Lasso': {'model__alpha': np.linspace(0.01, 10.0, 100),
                  'model__max_iter': [100, 500, 1000, 5000, 10000]},
        'Random Forest Regressor': {
            'model__n_estimators': [100, 200, 300, 400, 500],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]},
        'Decision Tree Regressor': {'model__max_depth': [None, 10, 20, 30],
                                    'model__min_samples_split': [2, 5, 10],
                                    'model__min_samples_leaf': [1, 2, 4]},
        'XGBoost Regressor': {'model__n_estimators': [100, 200, 300, 400, 500],
                              'model__learning_rate': [0.01, 0.1, 0.2, 0.3],
                              'model__max_depth': [3, 5, 7, 9],
                              'model__subsample': [0.6, 0.8, 1.0],
                              'model__colsample_bytree': [0.6, 0.8, 1.0]},
        'AdaBoost Regressor': {'model__n_estimators': [50, 100, 150, 200],
                               'model__learning_rate': [0.01, 0.1, 0.5, 1.0],
                               'model__loss': ['linear', 'square',
                                               'exponential']},
        'Gradient Boosting Regressor': {
            'model__n_estimators': [100, 200, 300, 400, 500],
            'model__learning_rate': [0.01, 0.1, 0.2, 0.3],
            'model__max_depth': [3, 5, 7, 9],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]}
    }

    # Sidebar - Trigger predictions
    if st.sidebar.button('Predict Prices'):
        if target in df.columns and features:
            X = filtered_df[features]
            y = filtered_df[target]

            if len(X) < 100:
                st.warning(
                    "Insufficient data, generating synthetic data to ensure enough samples.")
                if y.nunique() <= 1:
                    noise = np.random.normal(0, 0.01, y.shape)
                    y_noisy = y + noise
                    X_encoded = preprocessor.fit_transform(X)
                    df_encoded = pd.DataFrame(X_encoded,
                                              columns=[f"col_{i}" for i in
                                                       range(X_encoded.shape[
                                                                 1])])
                    df_encoded[target] = y_noisy

                    k_neighbors = min(5,
                                      len(df_encoded) - 1)  # Set k_neighbors to a value less than the number of observations
                    try:
                        # phi_relevance = {'control.pts': [[min(y_noisy), 1],
                        #                                  [max(y_noisy), 1]],
                        #                  'power': 1, 'relevance.type': 'phi'}
                        # df_resampled = smogn.smoter(
                        #     data=df_encoded,
                        #     y=target,
                        #     k=k_neighbors,
                        #     samp_method="balance",
                        #     rel_thres=0.01,
                        # )
                        df_resampled = df_encoded
                        X_resampled = df_resampled.drop(columns=[target])
                        y_resampled = df_resampled[target]
                    except ValueError as e:
                        st.error(f"Error during SMOGN: {e}")
                        X_resampled, y_resampled = X, y_noisy
                else:
                    X_resampled, y_resampled = resample(X, y, replace=True,
                                                        n_samples=100,
                                                        random_state=42)
            else:
                X_resampled, y_resampled = X, y

            # Ensure X_resampled and y_resampled have at least 10 rows
            while len(X_resampled) < 10:
                X_resampled, y_resampled = resample(X, y, replace=True,
                                                    n_samples=10,
                                                    random_state=42)
                # Add noise to numeric columns only
                numeric_X_resampled = X_resampled.select_dtypes(
                    include=[np.number])
                numeric_X_resampled += np.random.normal(0, 0.01,
                                                        numeric_X_resampled.shape)
                X_resampled[numeric_X_resampled.columns] = numeric_X_resampled
                y_resampled += np.random.normal(0, 0.01, y_resampled.shape)

            # Logging
            logging.info(f"X_resampled: {X_resampled}")
            logging.info(f"y_resampled: {y_resampled}")

            X_train, X_test, y_train, y_test = train_test_split(X_resampled,
                                                                y_resampled,
                                                                test_size=0.2,
                                                                random_state=42)

            model = model_dict[model_type]
            pipeline = Pipeline(
                [('preprocessor', preprocessor), ('model', model)])

            if search_type == "GridSearch":
                search = GridSearchCV(pipeline,
                                      param_distributions[model_type],
                                      cv=cv_folds, verbose=3)
            elif search_type == "RandomizedSearch":
                search = RandomizedSearchCV(pipeline,
                                            param_distributions[model_type],
                                            n_iter=n_iter_search, cv=cv_folds,
                                            random_state=42, verbose=3)
            else:
                search = pipeline

            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time = time()
            console_output = st.empty()

            # Fit the model and log progress
            search.fit(X_train, y_train)
            progress_bar.progress(100)
            status_text.text("Training complete!")
            end_time = time()

            # Model performance metrics
            y_pred = search.predict(X_test)

            st.header('Model Performance')
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f'Mean Squared Error (MSE): {mse}')
            st.write(f'Mean Absolute Error (MAE): {mae}')
            st.write(f'R^2 Score: {r2:.4f}')
            st.write('---')
            st.write(f'Predicted mean price: {np.mean(y_pred)}')

            # Best hyperparameters
            if search_type in ["GridSearch", "RandomizedSearch"]:
                st.write("Best Hyperparameters:")
                st.write(search.best_params_)

            # Visualization of Actual vs Predicted
            fig = px.scatter(x=y_test, y=y_pred,
                             labels={'x': 'Actual', 'y': 'Predicted'},
                             title='Actual vs Predicted')
            fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                     y=[y_test.min(), y_test.max()],
                                     mode='lines',
                                     line=dict(color='red', dash='dash'),
                                     name='Ideal Fit'))
            st.plotly_chart(fig)

            # Feature importance or model coefficients visualization
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = model.coef_
            else:
                importance = [0] * len(
                    features)  # Default to zeros if no importance or coefficients

            # Ensure features and importance have the same length
            if len(features) != len(importance):
                st.warning(
                    f"Length mismatch: features({len(features)}) and importance({len(importance)})")
                feature_importance_df = pd.DataFrame(
                    {'Features': features[:len(importance)],
                     'Importance': importance}).sort_values(by='Importance',
                                                            ascending=False)
            else:
                feature_importance_df = pd.DataFrame({'Features': features,
                                                      'Importance': importance}).sort_values(
                    by='Importance', ascending=False)

            # Discard the least important variables to match the lengths
            feature_importance_df = feature_importance_df.sort_values(
                by='Importance', ascending=True).head(len(importance))

            features = feature_importance_df['Features'].tolist()
            importance = feature_importance_df['Importance'].tolist()

            fig_importance = px.bar(x=features, y=importance,
                                    labels={'x': 'Features',
                                            'y': 'Importance or Weight'},
                                    title='Feature Importance or Weights')
            st.plotly_chart(fig_importance)

            # Feature importance table with download option
            st.write("Feature Importances")
            st.dataframe(feature_importance_df)
            csv = feature_importance_df.to_csv().encode('utf-8')
            st.download_button(label="Download Feature Importances", data=csv,
                               file_name='feature_importances.csv',
                               mime='text/csv')
        else:
            st.error(
                "Please select a valid target variable and at least one feature for prediction.")
