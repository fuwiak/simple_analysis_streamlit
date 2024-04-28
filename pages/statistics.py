import pandas as pd
import numpy as np
import streamlit as st
import pandas_profiling
from pandas_profiling import ProfileReport

from streamlit_pandas_profiling import st_profile_report


st.write('''# Статистический анализ данных''')

path ='merged_leads_land_not_null.csv'
df = pd.read_csv(f"data/{path}") #загрузка данных

pr = df.profile_report()
st_profile_report(pr)

# ProfileReport(df, explorative=True)









