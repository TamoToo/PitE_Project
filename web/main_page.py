import streamlit as st
import bitfinex
from datetime import datetime
import time
from PIL import Image

import sys
sys.path.append('../')
from source import module


# Web Icon
darius = Image.open("imgs/darius.gif")
st.set_page_config(page_title="Crypto Price Prediction", page_icon=darius)

st.markdown("# Main page 🎈")
st.sidebar.markdown("# Main page 🎈")

_left, _mid, _right = st.columns(3)
with _mid:
    st.write("[![Star](<https://img.shields.io/github/stars/TamoToo/PitE_Project.svg?logo=github&style=social>)](https://github.com/TamoToo/PitE_Project)")
    st.image("imgs/fuse.gif", use_column_width=True)
with _right:
    st.image("imgs/eth.gif", use_column_width=True)
with _left:
    st.image("imgs/btc.gif", use_column_width=True)


st.title("Cryptocurrency Price Prediction!")

# Showing all pairs
st.header("List of pairs available")
api_v1 = bitfinex.bitfinex_v1.api_v1()
pairs = api_v1.symbols()
st.write(pairs)

# Search for a specific pair
st.header("Search a pair with first letters")
check = str.lower(st.text_input("Enter the begening of the pair your are looking for", "btc"))
res = [idx.strip('"') for idx in pairs if idx.startswith(check)]
st.write(res)

pair = str.lower(st.text_input("Enter the pair to use", "btcusd"))

# Check if there is only one result possible with the pair wanted
if not pair in pairs:
    st.write("The pair you entered is not available")
    st.write("did you mean")
    re = [idx for idx in pairs if idx.startswith(pair)]
    st.write(re)
    st.stop()



st.write("You have chosen the pair: ", pair)

# Chose the interval
st.header("Interval")
st.write("Please choose the interval of the data you want to use")
st.write("Please be aware that if you choose a short interval, the function may take a long time.")
## We set a default value at 1D for the interval to avoid runtime problems
values = ["1m","5m", "15m", "30m", "1h", "3h", "6h", "12h", "1D", "7D", "14D", "1M"]
default_ix = values.index("1D")
interval = st.selectbox('Choose the interval', values, index=default_ix)
st.subheader("You have chosen the interval: ", interval)

# Chose start date
st.header("Start Date")
st.write("Please choose the start date of the data you want to use")
start = st.date_input("Start date", datetime(2019, 1, 1))
st.write("You have chosen the start date: ", start)

# Chose end date
st.header("End Date")
st.write("Please choose the end date of the data you want to use")
end = st.date_input("End date", datetime.now())
st.write("You have chosen the end date: ", end)

# Get the data from bitfinex API
df = module.fetch_data(start=time.mktime(start.timetuple()) * 1000, stop=time.mktime(end.timetuple()) * 1000, symbol=pair, interval=interval)
st.dataframe(df)

# Chose if we want to reduce the dataframe
st.write("Please choose the number of the data you want to use")
nb_data = st.slider("Choose the number of data you want to use", min_value=0, max_value=len(df), value=(0, len(df)))
st.write("You have chosen the number of data: ", nb_data[0], " - ", nb_data[1])
df = df.iloc[nb_data[0]:nb_data[1]]
st.dataframe(df)
st.write("The dataframe has been created")
    
# Chose prediction start date 
st.subheader("Prediction start date")
st.write("Please select the date when you want to start the prediction")
start_prediction = st.date_input("Start prediction", min_value=start, max_value=datetime.now())
st.write("You have chosen the start date: ", start_prediction)
