# Load the Models
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle

# Read the models
with open('model_lr_pipe2.pkl', 'rb') as file_1:
    model_lr_pipe2=pickle.load(file_1)  
    
with open('scaler.pkl', 'rb') as file_2:
    scaler=pickle.load(file_2)


def run():
    with st.form(key='Number of Weeks'):
        week = st.slider('Weeks', 1,20,4,help='Number of Weeks You Want To Predict')
        
        st.markdown('---')
        submitted = st.form_submit_button('Predict')

    # Read the original data
    df = pd.read_csv('sample_dataset_timeseries_noarea.csv')
    df['week_start_date']=pd.to_datetime(df['week_start_date'])
    df['week_end_date']=pd.to_datetime(df['week_end_date'])
    week_quantity=df.groupby(['week_start_date'])['quantity'].sum().reset_index(name='quantity')
    week_quantity_ts=week_quantity[['week_start_date', 'quantity']][1:].set_index('week_start_date')

    if submitted:
        # Define the forecasting model
        def forecasting(week):
        # :param `week` : how many weeks to predict
            week_quantity_ts_forecast = week_quantity_ts.squeeze().copy()
            window = 2
            for i in range(week):
                X = np.array(week_quantity_ts_forecast[-window:].values).reshape(1, -1)
                X_scaled = scaler.transform(X)
                new_idx=week_quantity_ts_forecast.index[-1]+datetime.timedelta(days=7)
                week_quantity_ts_forecast[new_idx] = round(model_lr_pipe2.predict(X_scaled)[0])
            return week_quantity_ts_forecast
        
        # Forecast for the Next 10 weeks
        quantity_forecast = forecasting(week).to_frame()
        st.write("The Forecasted Quantity")
        st.dataframe(quantity_forecast['quantity'][66:])

        # Forecasting 10 weeks ahead from real data
        fig, ax = plt.subplots(nrows=1,figsize=(10,3))
        ax.plot(quantity_forecast)
        ax.plot(week_quantity_ts['quantity'])
        ax.set_title('Forecasting using Linear Regression')
        ax.set_xlabel('Date')
        ax.set_ylabel('Quantity')
        ax.legend(['Forecasted Quantity', 'Original Quantity'])
        ax.grid()
        st.pyplot(fig)
        st.write('#### It is predicted that in the next',week,'weeks, the number of products sold will be in the range of '
                 ,quantity_forecast['quantity'][66:].min(),' and ',quantity_forecast['quantity'][66:].max(),
                 ' per week (with an error of +/- 510K).')
        
if __name__ == '__main__':
    run()