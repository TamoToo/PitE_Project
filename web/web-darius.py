import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import bitfinex
from datetime import datetime
import time
from PIL import Image


# ML methods
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error


import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from keras.callbacks import LearningRateScheduler
from keras.metrics import RootMeanSquaredError
from sklearn.linear_model import LinearRegression

import streamlit as st


darius=Image.open("darius.gif")
st.set_page_config(page_title="Darius Crypto Prediction", page_icon=darius)

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
    dataframe['time'] = pd.to_datetime(dataframe['time'], unit='ms')
    dataframe = dataframe.sort_values(by='time')
    dataframe.reset_index(inplace=True)
    dataframe.drop('index', axis=1, inplace=True)
    dataframe.rename(columns={'time':'date'}, inplace=True)
    return dataframe

st.sidebar.markdown("### The Darius description ")
st.sidebar.markdown("Darius' attacks and damaging abilities cause enemies to bleed for physical damage over 5 seconds, stacking up to 5 times. Darius enrages and gains massive Attack Damage when his target reaches max stacks.")

_left, mid, _right = st.columns(3)
with mid:
   st.image("fuse.gif", use_column_width=True)
with _left:
   st.image("ethe.gif", use_column_width=True)

with _right:
   st.image("bite.gif", use_column_width=True)



st.title("Cryptocurrency Darius Prediction!")

## if user has a dataset, show it
uploaded_file = st.file_uploader("Choose a file or select the code of the crypto to import it")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)
else:
    st.write("List of pairs available")
    api_v1 = bitfinex.bitfinex_v1.api_v1()
    pairs = api_v1.symbols()
    st.write(pairs)

    st.write("Search a pair with first letters")
    check = str.lower(st.text_input("Enter the begening of the pair your are looking for", "btc"))
    res = [idx for idx in pairs if idx.startswith(check)]
    st.write(res)

    pair = str.lower(st.text_input("Enter the pair to use", "btcusd"))

    ##check if there is only one result possible with the pair wanted
    if pair in pairs:
        find=True
    else:
        find=False
        st.write("The pair you entered is not available")
        st.write("did you mean")
        re=[idx for idx in pairs if idx.startswith(pair)]
        st.write(re)
    
    if find:
        st.write("You have chosen the pair: ", pair)
        st.write("Please choose the interval of the data you want to use")
        interval = st.selectbox("Choose the interval", ["1m", "5m", "15m", "30m", "1h", "3h", "6h", "12h", "1D", "7D", "14D", "1M"])
        st.write("You have chosen the interval: ", interval)
        st.write("Please choose the start date of the data you want to use")
        start = st.date_input("Start date", datetime(2019, 1, 1))
        st.write("You have chosen the start date: ", start)
        st.write("Please choose the end date of the data you want to use")
        end = st.date_input("End date", datetime.now())
        st.write("You have chosen the end date: ", end)

        df = fetch_data(start=time.mktime(start.timetuple()) * 1000, stop=time.mktime(end.timetuple()) * 1000, symbol=pair, interval=interval)
        st.write(df)
        st.write("The dataframe has been created")
        st.write("Please choose the number of the data you want to use")
        nb_data = st.slider("Choose the number of data you want to use", min_value=1, max_value=len(df), value=len(df))
        st.write("You have chosen the number of data: ", nb_data)
        df = df.iloc[-nb_data:]
        st.write(df)
        st.write("The dataframe has been created")
        
        #t_start = datetime(2009, 1, 1, 0, 0)
        #t_start = time.mktime(t_start.timetuple()) * 1000
        #t_stop = datetime(2023, 1, 1, 0, 0)
        #t_stop = time.mktime(t_stop.timetuple()) * 1000
        #df = fetch_data(start=t_start, stop=t_stop, symbol=pair)
        #st.write(df)
        #st.write("The dataframe has been created")



#pair = str.lower(st.text_input("Enter the pair to use", "btcusd"))
#st.title(pair)

##intervals = st.text_input("Enter the number of day for the interval", "1D ,2D ,3D")
#t_start = datetime(2009, 1, 1, 0, 0)
#t_start = time.mktime(t_start.timetuple()) * 1000
#t_stop = datetime(2023, 1, 1, 0, 0)
#t_stop = time.mktime(t_stop.timetuple()) * 1000
#df = fetch_data(start=t_start, stop=t_stop, symbol=pair)
#st.write(df)
