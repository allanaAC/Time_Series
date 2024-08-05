import pandas as pd
import logging

def create_features(data):
    try:
        data['Next_day'] = data['Close'].shift(-1)
        data['Target'] = (data['Next_day'] > data['Close']).astype(int)
        return data
    except Exception as e:
        logging.error(" Error in create_features data: {}". format(e))