import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.append('../')
sys.path.append('../../')
from source import module
from main_page import start_prediction, df


# Web Icon
darius = Image.open("imgs/darius.gif")
st.set_page_config(page_title="Crypto Price Prediction", page_icon=darius)

st.markdown("# Forecast 📈")
st.sidebar.markdown("# Forecast 📈")

try:
    X, y, index = module.split_data(df, str(start_prediction))
except:
    st.write("Please choose a date within the range you chose")
    st.stop()

# Scaling the data
y = y.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(y)
y = scaler.transform(y)

# Sliders for lookback / forecast periods
n_lookback = st.slider("Lookback period", min_value=0, max_value=len(df), value=30)
n_forecast = st.slider("Forecast period", min_value=0, max_value=30, value=10)


X_forecast = []
y_forecast = []

for i in range(n_lookback, len(y) - n_forecast + 1):
    X_forecast.append(y[i - n_lookback: i])
    y_forecast.append(y[i: i + n_forecast])

X_forecast = np.array(X_forecast)
y_forecast = np.array(y_forecast)

# LSTM options
lstm_neurons = 100
epochs = 50
batch_size = 32
loss = 'mse'
dropout = 0.2
optimizer = 'adam'


# fit the model
forecast_model = module.build_lstm_model(X_forecast, output_size=n_forecast, neurons=lstm_neurons, dropout=dropout, loss=loss, optimizer=optimizer)
forecast_model.fit(X_forecast, y_forecast, epochs=20, batch_size=32, verbose=0)

# generate the forecasts
X_ = y[- n_lookback:]  # last available input sequence
X_ = X_.reshape(1, n_lookback, 1)

y_ = forecast_model.predict(X_).reshape(-1, 1)
y_ = scaler.inverse_transform(y_)

# organize the results in data frames
df_past = df[['date', 'close']]
df_past['Forecast'] = np.nan
df_past['Forecast'].iloc[-1] = df_past['close'].iloc[-1]

join_dfs = pd.DataFrame({'date': df_past.iloc[-1]['date'], 'close': df_past.iloc[-1]['close'], 'Forecast': df_past.iloc[-1]['Forecast']}, index=[0])
df_future = pd.DataFrame(columns=['date', 'close', 'Forecast'])
df_future['date'] = pd.date_range(start=df_past['date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
df_future['Forecast'] = y_.flatten()
df_future['close'] = np.nan
df_future = pd.concat([join_dfs, df_future], axis=0).reset_index(drop=True)

# plot the results
fig, ax = plt.subplots(figsize=(40,20))
ax.plot(df_past['date'][-n_lookback:], df_past['close'][-n_lookback:])
ax.plot(df_future['date'], df_future['Forecast'], color='red')
ax.legend(['past price', 'forecast'], prop={'size': 42})
st.pyplot(fig)