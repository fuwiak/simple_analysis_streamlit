import pandas as pd
import numpy as np
import streamlit as st
import mpld3
import streamlit.components.v1 as components
import matplotlib.pylab as plt
import seaborn as sns


st.title("Визуализация данных")

df = pd.read_excel('/Users/user/PycharmProjects/simple_analysis_streamlit/data/data.xlsx')

#Распределение целевой переменной
st.write('''## Распределение целевой переменной''')


col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(20, 10))
    cat_cols = df.select_dtypes(include=['object']).columns
    col = st.selectbox('Выберите категориальную переменную', cat_cols)
    sns.countplot(y=col, data=df, ax=ax)
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(20, 10))
    num_cols = df.select_dtypes(include=['number']).columns
    col = st.selectbox('Выберите числовую переменную', num_cols)
    sns.distplot(df[col], ax=ax)
    st.pyplot(fig)




# sns.countplot(x=col, data=df, ax=ax)
# fig_html = mpld3.fig_to_html(fig)
# components.html(fig_html, height=1000, width=1000)
#
#
# #Распределение числовых переменных
# # st.write('''## Распределение числовых переменных''')
#
# #select the column only numerical columns
# num_cols = df.select_dtypes(include=['number']).columns
# #histogram
# col = st.selectbox('Выберите числовую переменную', num_cols)
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111)
# ax.grid(False)
# sns.histplot(x=col, data=df, ax=ax)
# fig_html = mpld3.fig_to_html(fig)
# components.html(fig_html, height=1000, width=1000)
