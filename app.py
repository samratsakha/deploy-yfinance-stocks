# Required Libraries
from flask import Flask, render_template, request, make_response
import jsonify
import requests
import json
from requests.sessions import Request
import pickle
import numpy as np
# Model building requirements
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
import pandas as pd
import yfinance as yf
import datetime 



app = Flask(__name__)



# prepare data for model
def prepare_data(timeseries_data, n_features):
	X, y =[],[]
	for i in range(len(timeseries_data)):
		end_ix = i + n_features
		if end_ix > len(timeseries_data)-1:
			break
		seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)


# recieve data from api
def recieve_data(ticker_name):
    start = datetime.datetime(2020,6,21) 
    data = yf.Ticker(ticker_name) 
    df = data.history(start=start)
    return df


# preprocess data
def preprocess_data(df):

    df = df.reset_index()
    data = df['Open'].copy()
    data = data.dropna(axis=0)

    timeseries_data = data.to_list()
    n_steps = 10
    X, y = prepare_data(timeseries_data, n_steps)

    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    return X,y,timeseries_data[len(data)-10:len(data)]




# model building and prediction
def build_model(X,y,x_input):

    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(10, 1)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    model.fit(X, y, epochs=100, verbose=1)

    

    # prepare data for newer predictions
    x_input = np.array(x_input)
    temp_input=list(x_input)
    lst_output=[]
    i=0
    while(i<10):
        
        if(len(temp_input)>10):
            x_input=np.array(temp_input[1:])
            x_input = x_input.reshape((1, 10, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.append(yhat[0][0])
            temp_input=temp_input[1:]
            lst_output.append(yhat[0][0])
            i=i+1
        else:
            x_input = x_input.reshape((1, 10, 1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.append(yhat[0][0])
            lst_output.append(yhat[0][0])
            i=i+1

    return lst_output
        



# Templates
# Home page
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')



# API JSON
@app.route("/to_model", methods=['POST'])
def to_model():

    req = request.get_json()
    stock_name = req['val_array']

    try:

        df = recieve_data(stock_name)

        if (len(df)<100):
            outs = "WRONG"
            x = {"output": outs}
            y = json.dumps(x)
            return y

        
        X,y,test_data = preprocess_data(df)
        preds = build_model(X,y,test_data)


        outs = str(preds)

        x = {"output": outs}
        y = json.dumps(x)

        return y

    except:
        x = {"output": "EXCEPTION"}
        y = json.dumps(x)

        return y



if __name__=="__main__":
    app.run(debug=True)

