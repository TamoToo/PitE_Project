import streamlit as st
import sys
sys.path.append('../')
sys.path.append('../../')
from source import module
from streamlit_app import start_prediction, df

import numpy as np
from sklearn.preprocessing import MinMaxScaler

st.write(start_prediction)

try:
    X, y, index = module.split_data(df, str(start_prediction))
except:
    st.write("Please choose a date within the range you chose")
    st.stop()


st.header("LSTM")

X_scaler = MinMaxScaler(feature_range=(0,1))
y_scaler = MinMaxScaler(feature_range=(0,1))

X = X_scaler.fit_transform(np.array(X))
y = y_scaler.fit_transform(np.array(y).reshape(-1,1))

X_train = X[:index]
y_train = y[:index]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_test = X[index:]
y_test = y[index:]
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


lstm_neurons = 100
epochs = 50
batch_size = 32
loss = 'mse'
dropout = 0.2
optimizer = 'adam'


lstm_model = module.build_lstm_model(X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss, optimizer=optimizer)
hist = lstm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1, shuffle=False)

# make predictions
lstm_train_prediction = lstm_model.predict(X_train)
lstm_test_prediction = lstm_model.predict(X_test)

# invert predictions
## create empty table with 4 fields
lstm_train_prediction_dataset_like, lstm_test_prediction_dataset_like = np.zeros(shape=(len(lstm_train_prediction), X_train.shape[1])), np.zeros(shape=(len(lstm_test_prediction), X_test.shape[1]))
## put the predicted values in the right field
lstm_train_prediction_dataset_like[:,0], lstm_test_prediction_dataset_like[:,0] = lstm_train_prediction[:,0], lstm_test_prediction[:,0]
## inverse transform and then select the right field
lstm_train_prediction, lstm_test_prediction = X_scaler.inverse_transform(lstm_train_prediction_dataset_like)[:,0], X_scaler.inverse_transform(lstm_test_prediction_dataset_like)[:,0]
lstm_train_prediction, lstm_test_prediction = lstm_train_prediction.reshape(-1,1), lstm_test_prediction.reshape(-1,1)

y_train = y_scaler.inverse_transform(y_train)
y_test = y_scaler.inverse_transform(y_test)
module.show_errors(y_train, y_test, lstm_train_prediction, lstm_test_prediction)

train_size = X_train.shape[0]
module.plot_results(df, train_size, lstm_test_prediction)