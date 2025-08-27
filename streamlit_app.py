import streamlit as st

st.title(' Prediction Factor of Safety')

st.info('This is app create prediction using soil nailing !!')

# Step 1: Load Data
with st.expander('Data'):
  st.write('Raw Data')
  df = pd.read_csv('https://raw.githubusercontent.com/farishadi18-stack/fh-predictionslopestability/refs/heads/master/new%20treated%20slope.csv')
  df
