import streamlit as st
import eda
import prediction

navigation = st.sidebar.selectbox('Choose Page : ', ('Forecast The Quantities Sold','EDA'))

if navigation == 'EDA':
    eda.run()
else:
    prediction.run()