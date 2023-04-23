import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from datetime import date
# from keras.models import load_model
import streamlit as st
import yfinance as yf

from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
START = '2010-01-01'
TODAY = date.today().strftime("%Y-%m-%d")


st.title('Stock Trend Prediction')
stocks = ("AAPL", "GOOG", "MSFT")
user_input = st.text_input("Enter Stock Ticker", 'AAPL')

n_years = st.slider("Years of Prediction:", 1, 4)
period = n_years*365


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text("Load Data...")
data = load_data(user_input)

data_load_state.text("Loading data .. done!")

st.subheader('Raw Data')
st.write(data.tail())

# st.subheader('Closing Price vs Time Chart')
# fig = plt.figure(figsize=(12, 6))
# plt.plot(data.Close)
# st.pyplot(fig)

# st.subheader('Closing Price vs Time Chart with 100MA')
# ma100 = data.Close.rolling(100).mean()
# fig = plt.figure(figsize=(12, 6))
# plt.plot(ma100)
# st.pyplot(fig)

# st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
# ma100 = data.Close.rolling(100).mean()
# ma200 = data.Close.rolling(200).mean()
# fig = plt.figure(figsize=(12, 6))
# plt.plot(ma100)
# plt.plot(ma200)
# st.pyplot(fig)


# data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
# data.testing = pd.DataFrame(data['Close'][int(len(data)*0.70):int(len(data))])

# print(data_training.shape)
# print(data.testing.shape)

# scaler = MinMaxScaler(feature_range=(0, 1))

# data_training_array = scaler.fit_transform(data_training)

# x_train = []
# y_train = []

# for i in range(100, data_training_array.shape[0]):
#     x_train.append(data_training_array[i-100:i])
#     y_train.append(data_training_array[i, 0])

# x_train, y_train = np.array(x_train), np.array(y_train)

# model = load_model('keras_model')
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],
                  y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'],
                  y=data['Close'], name="stock_close"))
    fig.layout.update(
        title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()


df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
