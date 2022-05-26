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
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from keras.callbacks import LearningRateScheduler
from keras.metrics import RootMeanSquaredError
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

import streamlit as st


darius=Image.open("darius.gif")
st.set_page_config(page_title="Darius Crypto Prediction", page_icon=darius)


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
        st.write(df)
        st.write("Please choose the number of the data you want to use")
        nb_data = st.slider("Choose the number of data you want to use", min_value=0, max_value=len(df), value=(0, len(df)))
        st.write("You have chosen the number of data: ", nb_data[0], " - ", nb_data[1])
        df = df.iloc[nb_data[0]:nb_data[1]]
        st.write(df)
        st.write("The dataframe has been created")
        



## function to create datasplit depending of the date starting the prediction
X_scaler = MinMaxScaler(feature_range=(0,1))
y_scaler = MinMaxScaler(feature_range=(0,1))


st.header("Linear Regression")
st.subheader("Prediction start date")
st.write("Please select the date when you want to start the prediction")
start_prediction = st.date_input("Start prediction", min_value=start, max_value=datetime.now())
st.write("You have chosen the start date: ", start_prediction)

try:
    X_train, y_train, X_test, y_test = module.split_data(df, str(start_prediction))
except:
    st.write("Please choose a date within the range you chose")
    st.stop()

linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)
st.write("Fitting the model...")


#first we predict using train set and then we use the test set to evaluate the model
linear_regression_train_predict=linear_regression_model.predict(X_train)
linear_regression_validation_predict=linear_regression_model.predict(X_test)
st.write("The linear regression model has been trained...")
st.write("The linear regression model has been evaluated...")
st.write("Train data prediction shape: ", linear_regression_train_predict.shape)
st.write("Validation data prediction shape: ", linear_regression_validation_predict.shape)
st.write("Mean Absolute Error - MAE :", mean_absolute_error(y_test, linear_regression_validation_predict))

train_size = X_train.shape[0]
module.plot_results(df, train_size, linear_regression_validation_predict)

lr_gs_model = LinearRegression()

# parameters that we will try to tune
params_lr_gs = {
    'n_jobs': range(1, 1000),
}

param_search = GridSearchCV( estimator=lr_gs_model, param_grid=params_lr_gs,
                verbose=1)
                
param_search.fit(X_train, y_train)

best_score = param_search.best_score_
best_params = param_search.best_params_

st.write("The best score is: ", best_score, "with the following parameters: ", best_params)

lr_final_model = LinearRegression(**best_params)
lr_final_model.fit(X_train, y_train)
lr_final_train_predict=lr_final_model.predict(X_train)
lr_final_validation_predict=lr_final_model.predict(X_test)

st.write("Train data prediction shape: ",lr_final_train_predict.shape)
st.write("Validation data prediction shape: ", lr_final_validation_predict.shape)
st.write("Mean Absolute Error - MAE :", mean_absolute_error(y_test, lr_final_validation_predict))

module.plot_results(df, train_size, lr_final_validation_predict)
