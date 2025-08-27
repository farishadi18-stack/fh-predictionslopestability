import streamlit as st

st.title(' Prediction Factor of Safety')

st.info('This is app create prediction using soil nailing !!')

# Step 1: Load Data
with st.expander('Data'):
  st.write('Raw Data')
  df = pd.read_csv('https://raw.githubusercontent.com/ArifAzhar243/artificialneuralnetworkbyaa/refs/heads/master/aa%20Machine%20Learning.csv')
  df
