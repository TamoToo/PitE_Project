import streamlit as st
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

import sys
sys.path.append('../')
sys.path.append('../../')
from source import module
from main_page import start_prediction, df


# Web Icon
darius = Image.open("imgs/darius.gif")
st.set_page_config(page_title="Crypto Price Prediction", page_icon=darius)

st.markdown("# Linear Regression 🪄")
st.sidebar.markdown("# Linear Regression 🪄")

# Write previous informations
st.write(start_prediction)
st.dataframe(df)

# Split the data
try:
    X, y, index = module.split_data(df, str(start_prediction))
except:
    st.write("Please choose a date within the range you chose")
    st.stop()

# Preparing the data
X_train = X[:index]
y_train = y[:index]
X_test = X[index:]
y_test = y[index:]

# Fit / Predict
lr_model = LinearRegression(n_jobs=1)
lr_model.fit(X_train, y_train)
lr_train_prediction = lr_model.predict(X_train)
lr_test_prediction = lr_model.predict(X_test)

st.write("The linear regression model has been trained...")
st.write("The linear regression model has been evaluated...")

# Show errors (MSE / MAE...)
module.show_errors(y_train, y_test, lr_train_prediction, lr_test_prediction)

# Plotting results
train_size = X_train.shape[0]
module.plot_results(df, train_size, lr_test_prediction)