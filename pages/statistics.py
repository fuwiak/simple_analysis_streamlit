import pandas as pd
import numpy as np
import streamlit as st
import pandas_profiling
from pandas_profiling import ProfileReport

from streamlit_pandas_profiling import st_profile_report


st.write('''# Статистический анализ данных''')

df = pd.read_excel('/Users/user/PycharmProjects/simple_analysis_streamlit/data/data.xlsx')

pr = df.profile_report()
st_profile_report(pr)

# ProfileReport(df, explorative=True)









