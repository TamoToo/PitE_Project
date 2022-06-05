import streamlit as st
import sys
sys.path.append('../')
sys.path.append('../../')
from source import module
from streamlit_app import start_prediction, df

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

st.write(start_prediction)
st.write(df)

try:
    X, y, index = module.split_data(df, str(start_prediction))
except:
    st.write("Please choose a date within the range you chose")
    st.stop()


st.header("Linear Regression")


X_train = X[:index]
y_train = y[:index]
X_test = X[index:]
y_test = y[index:]

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

st.write("The linear regression model has been trained...")
st.write("The linear regression model has been evaluated...")
module.show_errors(y_train, y_test, lr_final_train_predict, lr_final_validation_predict)

train_size = X_train.shape[0]
module.plot_results(df, train_size, lr_final_validation_predict)