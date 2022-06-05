import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import bitfinex
from datetime import datetime
import time
from PIL import Image

# Functions
import sys
sys.path.append('../')
from source import module

# ML methodsP
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
from keras.callbacks import LearningRateScheduler
from keras.metrics import RootMeanSquaredError
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

import streamlit as st


darius=Image.open("darius.gif")
st.set_page_config(page_title="Crypto Price Prediction", page_icon=darius)


st.markdown("# Main page 🎈")
st.sidebar.markdown("# Main page 🎈")

_left, mid, _right = st.columns(3)
with mid:
   st.image("fuse.gif", use_column_width=True)
with _left:
   st.image("ethe.gif", use_column_width=True)

with _right:
   st.image("bite.gif", use_column_width=True)



st.title("Cryptocurrency Price Prediction!")

st.header("List of pairs available")
api_v1 = bitfinex.bitfinex_v1.api_v1()
pairs = api_v1.symbols()
st.write(pairs)

st.header("Search a pair with first letters")
check = str.lower(st.text_input("Enter the begening of the pair your are looking for", "btc"))
res = [idx.strip('"') for idx in pairs if idx.startswith(check)]
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
    st.header("Interval")
    st.write("Please choose the interval of the data you want to use")
    st.write("Please be aware that if you choose a short interval, the function may take a long time.")
    ## We set a default value at 1D for the interval
    values = ["1m","5m", "15m", "30m", "1h", "3h", "6h", "12h", "1D", "7D", "14D", "1M"]
    default_ix = values.index("1D")
    interval = st.selectbox('Choose the interval', values, index=default_ix)
    st.subheader("You have chosen the interval: ", interval)

    st.header("Start Date")
    st.write("Please choose the start date of the data you want to use")
    start = st.date_input("Start date", datetime(2019, 1, 1))
    st.write("You have chosen the start date: ", start)

    st.header("End Date")
    st.write("Please choose the end date of the data you want to use")
    end = st.date_input("End date", datetime.now())
    st.write("You have chosen the end date: ", end)

    df = module.fetch_data(start=time.mktime(start.timetuple()) * 1000, stop=time.mktime(end.timetuple()) * 1000, symbol=pair, interval=interval)
    st.dataframe(df)
    st.write("Please choose the number of the data you want to use")
    nb_data = st.slider("Choose the number of data you want to use", min_value=0, max_value=len(df), value=(0, len(df)))
    st.write("You have chosen the number of data: ", nb_data[0], " - ", nb_data[1])
    df = df.iloc[nb_data[0]:nb_data[1]]
    st.dataframe(df)
    st.write("The dataframe has been created")
        




st.subheader("Prediction start date")
st.write("Please select the date when you want to start the prediction")
start_prediction = st.date_input("Start prediction", min_value=start, max_value=datetime.now())
st.write("You have chosen the start date: ", start_prediction)



