import streamlit as st
import pandas as pd
import plotly.express as px


st.header('Карты отраслевой оценки')

df = pd.read_csv('pages/fin_data.csv', sep=';', index_col=[0])

df = df.astype('float')
df['Чистый долг / EBITDA'] = df['Чистый долг'] / df['EBITDA']
df['EV / EBITDA'] = df['EV'] / df['EBITDA']

st.write(df)


fig = px.scatter(df, x='EV / EBITDA', y='Чистый долг / EBITDA', color=df.index,
                 size='EV', hover_data=['EBITDA'])
st.plotly_chart(fig)

