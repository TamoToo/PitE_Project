import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import bitfinex
from datetime import datetime
import time
from math import sqrt
import streamlit as st

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score

@st.cache
def fetch_data(start=1640991600000, stop=1651356000000, symbol='btcusd', interval='1D'):
    print(start )
    # Create api instance
    api_v2 = bitfinex.bitfinex_v2.api_v2()

    # Defining intervals in ms
    intervals_dict = {"1m": 60000, "5m": 300000, "15m": 900000, "30m": 1800000, "1h": 3600000, "3h": 10800000, "6h": 21600000, "12h": 43200000, "1D": 86400000, "7D": 604800000, "14D": 1209600000, "1M": 2628000000}
    step = intervals_dict[interval] * 1000
    data = []
    names = ['time', 'open', 'close', 'high', 'low', 'volume']

    if stop > time.time() * 1000: # stop value can't be higher than datetime.now()
        stop = datetime.now()
        stop = time.mktime(stop.timetuple()) * 1000
    
    while stop - start > step: # while data requested > 1000 * interval
        if start + step > stop: # if start + 1000 * interval > stop ==> stop = now
            end = datetime.now()
            end = time.mktime(end.timetuple()) * 1000
        else:
            end = start + step
        #print(datetime.fromtimestamp(start / 1000), datetime.fromtimestamp(end / 1000))
        res = api_v2.candles(symbol=symbol, interval=interval, start=start, end=end)
        data.extend(res)
        start += step
        time.sleep(1)
    res = api_v2.candles(symbol=symbol, interval=interval, start=start, end=stop)
    data.extend(res)

    # Modify data to send back a clean DataFrame
    dataframe = pd.DataFrame(data, columns=names)
    dataframe['time'] = pd.to_datetime(dataframe['time'], unit='ms').dt.normalize()
    dataframe = dataframe.sort_values(by='time')
    dataframe.reset_index(inplace=True)
    dataframe.drop('index', axis=1, inplace=True)
    dataframe.rename(columns={'time':'date'}, inplace=True)

    return dataframe

def split_data(df, date):
    data = df.copy()
    data.reset_index(inplace=True)
    index = data.index[data['date'] == date][0]
    X = data.drop(columns=['index', 'date', 'close'], axis=1).to_numpy()
    y = data['close'].to_numpy()
    return X, y, index

def show_errors(y_train, y_test, train_prediction, test_prediction):
    st.write("Train data RMSE: ", sqrt(mean_squared_error(y_train,train_prediction)))
    st.write("Train data MSE: ", mean_squared_error(y_train,train_prediction))
    st.write("Train data MAE: ", mean_absolute_error(y_train,train_prediction))
    st.write("Train data explained variance regression score:", explained_variance_score(y_train, train_prediction))
    st.write("Train data R2 score:", r2_score(y_train, train_prediction))
    st.write("-------------------------------------------------------------------------------------")
    st.write("Test data RMSE: ", sqrt(mean_squared_error(y_test,test_prediction)))
    st.write("Test data MSE: ", mean_squared_error(y_test,test_prediction))
    st.write("Test data MAE: ", mean_absolute_error(y_test,test_prediction))
    st.write("Test data explained variance regression score:", explained_variance_score(y_test, test_prediction))
    st.write("Test data R2 score:", r2_score(y_test, test_prediction))

def plot_results(df, train_size, prediction):
    f,axs = plt.subplots(2,1,figsize=(40,20))

    axs[0].set_title("All time")
    axs[0].plot(df['date'][:train_size], df['close'][:train_size], color='black')
    axs[0].plot(df['date'][train_size:], df['close'][train_size:], color='green')
    axs[0].plot(df['date'][train_size:], prediction, color='red')
    axs[0].legend(['train', 'test', 'prediction'], prop={'size': 24})

    axs[1].set_title("Zoomed on prediction")
    axs[1].plot(df['date'][train_size:], df['close'][train_size:], color='green')
    axs[1].plot(df['date'][train_size:], prediction, color='red')
    axs[1].legend(['test', 'prediction'], prop={'size': 24})

    for ax in axs.flat:
        ax.set(xlabel="Time", ylabel="Price in $")

    st.pyplot(fig=f)



def build_lstm_model(input_data, output_size, neurons=100, activ_func='linear',
                     dropout=0.2, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model