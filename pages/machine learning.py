import pandas as pd
import numpy as np
import streamlit as st

import pandas as pd
import numpy as np
import streamlit as st
import mpld3
import streamlit.components.v1 as components
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



st.title("Обучение модели")

df = pd.read_excel('/Users/user/PycharmProjects/simple_analysis_streamlit/data/data.xlsx')

#выбор целевой переменной
st.write('''## Выбор целевой переменной''')
target_col = ['Входит в ядро РИНЦ']
# col1, col2 = st.columns(2)
# cat_cols =
col = st.selectbox('Выберите категориальную переменную', target_col)

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(20, 10))
    cat_cols = df.select_dtypes(include=['object']).columns
    sns.countplot(y=col, data=df, ax=ax)
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(20, 10))
    #value_counts as dataframe
    temp = df[col].value_counts().to_frame()
    temp.reset_index(inplace=True)
    temp.columns = [col, 'count']
    st.dataframe(temp)

#machine learning


options_num = st.multiselect(
    'Выберите числовые переменные для обучения модели',
    default=list(df.select_dtypes(include=['number']).columns)[1:3],
    options=list(df.select_dtypes(include=['number']).columns)

)



cat_cols = list(df.select_dtypes(include=['object']).columns)

remove_col = ['Автор', 'Входит в ядро РИНЦ','Название']
cat_cols = [x for x in cat_cols if x not in remove_col]




options_cat = st.multiselect(
    'Выберите категориальные переменные для обучения модели',
    default=cat_cols,
    options=cat_cols
)
#
#select columns

X = df[options_num + options_cat]
y = df[col]

#split data

#slider with step 0.05 from 0.1 to 0.5
test_size = st.slider('Выберите размер тестовой выборки', 0.1, 0.5, 0.2, 0.05)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
#label encoding
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


settings_map = {}
settings_map['test_size'] = test_size

#select type of encoding
encoding = st.selectbox(
    'Выберите тип кодирования категориальных переменных',
    ('OneHotEncoder','OneHotEncoder')
)

settings_map['encoding'] = encoding

#
#

#select type of scaling
scaling = st.selectbox(
    'Выберите тип масштабирования числовых переменных',
    ('MinMaxScaler', 'StandardScaler')
)
#
if scaling == 'MinMaxScaler':
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
else:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

settings_map['scaling'] = scaling

# #select model
#
model_name = st.selectbox(
    'Выберите модель для обучения',
    ('LogisticRegression', 'RandomForestClassifier')
)

settings_map['model_name'] = model_name

#predict
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, roc_curve, auc

def show_results(sorted_by_measure='accuracy'):
    test_size = settings_map['test_size']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=42)


    MLA_compare = pd.DataFrame()
    model = settings_map['model_name']
    encoding = settings_map['encoding']
    scaling = settings_map['scaling']
    if encoding == 'OneHotEncoder':
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder(handle_unknown='ignore')
        X_train = encoder.fit_transform(X_train)
        X_test = encoder.transform(X_test)
    if scaling == 'MinMaxScaler':
        from sklearn.preprocessing import MaxAbsScaler
        scaler = MaxAbsScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(with_mean=False)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    if model == 'LogisticRegression':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
    else:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()



    predicted = model.fit(X_train, y_train).predict(X_test)
    MLA_compare.loc[model_name, 'Accuracy'] = accuracy_score(y_test, predicted)
    MLA_compare.loc[model_name, 'Recall'] = recall_score(y_test, predicted)
    MLA_compare.loc[model_name, 'Precision'] = precision_score(y_test, predicted)
    MLA_compare.loc[model_name, 'F1'] = f1_score(y_test, predicted)
    MLA_compare.loc[model_name, 'MCC'] = matthews_corrcoef(y_test, predicted)
    return MLA_compare

st.write(show_results())


pip freeze > requirements.txt
