import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from moexalgo import Market, Ticker
import time

st.header('Карты отраслевой оценки')

df = pd.read_csv('pages/fin_data.csv', sep=';', index_col=[0])

#
# gazp = pd.DataFrame(Ticker('GAZP').marketdata()).loc[['LAST', 'ISSUECAPITALIZATION']]
# lkoh = pd.DataFrame(Ticker('LKOH').marketdata()).loc[['LAST', 'ISSUECAPITALIZATION']]
#
# merging = pd.concat([gazp, lkoh])
# st.write(merging)
df = df.astype('float')
df['Чистый долг / EBITDA'] = df['Чистый долг'] / df['EBITDA']
df['EV / EBITDA'] = df['EV'] / df['EBITDA']

st.write(df)


import plotly.express as px
# df = px.data.iris()
fig = px.scatter(df, x='EV / EBITDA', y='Чистый долг / EBITDA', color=df.index,
                 size='EV', hover_data=['EBITDA'])
st.plotly_chart(fig)

