import pandas as pd
import numpy as np
import yfinance as yf

import logging

#data_path = "/data/real_estate.csv"
def load_and_preprocess_data(file_path):   
    try:
        df = pd.read_csv(file_path)
        print(df.head())
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))
        
def preprocess_data(data):
    try:
        df = data.iloc[:-2, 0:2]
        df = df.set_index('Date')
        return df
    except Exception as e:
        logging.error(" Error in preprocess_data data: {}". format(e))
        
def load_yfinance_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        return data
    except Exception as e:
        logging.error(" Error in load_yfinance_data data: {}". format(e))