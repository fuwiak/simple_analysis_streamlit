import pandas as pd
import numpy as np
import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, ColumnsAutoSizeMode


st.set_page_config(page_title='Анализ и визуализация данных (научные публикации в области образования)', layout='wide')


st.title("Анализ и визуализация данных (научные публикации в области образования)")


path ='data/data.xlsx'
df = pd.read_excel(path) #чтение данных
#


st.markdown(''' # Данные ''')
st.dataframe(df)


# st.write("Название столбцов: ", df.columns)

col1, col2, col3, col4= st.columns(4)
with col1:
    st.write("Количество строк: ", df.shape[0])
    st.write("Количество столбцов: ", df.shape[1])
    st.write("Количество пустых значений: ", df.isnull().sum().sum())
    st.write("Количество дубликатов: ", df.duplicated().sum())
    st.write("Количество уникальных значений: ", df.nunique().sum())
    num_columns = df.select_dtypes(include=np.number).columns
    st.write("Количество числовых столбцов: ", len(num_columns))
    cat_columns = df.select_dtypes(include=np.object).columns
    st.write("Количество категориальных столбцов: ", len(cat_columns))



with col2:
    columns = pd.DataFrame(df.columns, columns=['Название столбцов'])
    st.dataframe(columns)


with col3:
    num_columns = df.select_dtypes(include=np.number).columns
    num_columns = pd.DataFrame(num_columns, columns=['Числовые столбцы'])
    st.dataframe(num_columns)

with col4:
    cat_columns = df.select_dtypes(include=np.object).columns
    cat_columns = pd.DataFrame(cat_columns, columns=['Категориальные столбцы'])
    st.dataframe(cat_columns)

# df info dataframe
df_info = pd.DataFrame(df.dtypes, columns=['Тип данных'])
df_info['Количество пустых значений'] = df.isnull().sum()
df_info['Количество уникальных значений'] = df.nunique()
df_info['Процент пустых значений'] = round(df.isnull().sum() / df.shape[0] * 100, 2)
df_info['Процент уникальных значений'] = round(df.nunique() / df.shape[0] * 100, 2)
# display dataframe with maximum width
st.dataframe(df_info, width=0)
