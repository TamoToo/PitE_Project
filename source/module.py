import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import bitfinex
from datetime import datetime
import time
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

def fetch_data(start=1640991600000, stop=1651356000000, symbol='btcusd', interval='1D', step=86400000):
    # Create api instance
    api_v2 = bitfinex.bitfinex_v2.api_v2()
    intervals_dict = {"1m": 60000, "5m": 300000, "15m": 900000, "30m": 1800000, "1h": 3600000, "3h": 10800000, "6h": 21600000, "12h": 43200000, "1D": 86400000, "7D": 604800000, "14D": 1209600000, "1M": 2628000000}
    step = intervals_dict[interval] * 1000
    data = []
    names = ['time', 'open', 'close', 'high', 'low', 'volume']

    if stop > time.time() * 1000: # stop value can't be higher than datetime.now()
        stop = datetime.now()
        stop = time.mktime(stop.timetuple()) * 1000
    if stop - start > step: # if data requested > 1000 * interval
        while start < stop:
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
    else:
        res = api_v2.candles(symbol=symbol, interval=interval, start=start, end=stop)
        data.extend(res)
    #print(data)

    # Modify data to send back a clean DataFrame
    dataframe = pd.DataFrame(data, columns=names)
    dataframe['time'] = pd.to_datetime(dataframe['time'], unit='ms').dt.normalize()
    dataframe = dataframe.sort_values(by='time')
    dataframe.reset_index(inplace=True)
    dataframe.drop('index', axis=1, inplace=True)
    dataframe.rename(columns={'time':'date'}, inplace=True)
    return dataframe

def split_data(data, date, lstm=0, X_scaler=MinMaxScaler(feature_range=(0,1)), y_scaler=MinMaxScaler(feature_range=(0,1))):
    data.reset_index(inplace=True)
    index = data.index[data['date'] == date][0]
    X = data.drop(columns=['index', 'date', 'close'], axis=1).to_numpy()
    y = data['close'].to_numpy()

    if lstm:
        X = X_scaler.fit_transform(np.array(X))
        y = y_scaler.fit_transform(np.array(y).reshape(-1,1))

        X_train = X[:index]
        y_train = y[:index]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        X_test = X[index:]
        y_test = y[index:]
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    else:
        X_train = X[:index]
        y_train = y[:index]
        X_test = X[index:]
        y_test = y[index:]

    return X_train, y_train, X_test, y_test


def plot_results(df, train_size, prediction):
    f,axs = plt.subplots(1,2,figsize=(40,20))

    axs[0].set_title("All time")
    axs[0].plot(df['date'][:train_size], df['close'][:train_size], color='black')
    axs[0].plot(df['date'][train_size:], df['close'][train_size:], color='green')
    axs[0].plot(df['date'][train_size:], prediction, color='red')
    axs[0].legend(['train', 'test', 'prediction'])

    axs[1].set_title("Zoomed on prediction")
    axs[1].plot(df['date'][train_size:], df['close'][train_size:], color='green')
    axs[1].plot(df['date'][train_size:], prediction, color='red')
    axs[1].legend(['test', 'prediction'])

    for ax in axs.flat:
        ax.set(xlabel="Time", ylabel="Price in $")

    st.pyplot(fig=f)