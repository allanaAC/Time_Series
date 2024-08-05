import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from statsmodels.tsa.arima.model import ARIMA
import pickle
from xgboost import XGBClassifier


# Function to train the model
def fit_arima_model(df):
    try:
        arima = ARIMA(df.AAPL, order=(1,1,1))
        ar_model = arima.fit()
        print(ar_model.summary())
        forecast = ar_model.get_forecast(2)
        return forecast
    except Exception as e:
        logging.error(" Error in fit_arima_model data: {}". format(e))
        
def fit_arimax_model(df):
    try:
        model2 = ARIMA(df.AAPL, exog=df.TXN, order=(1,1,1))
        arimax = model2.fit()
        print(arimax.summary())
        forecast = arimax.get_forecast(2)
        return forecast
    except Exception as e:
        logging.error(" Error in fit_arimax_model data: {}". format(e))
        
def train_xgboost_model(train, features):
    try:
        model = XGBClassifier(max_depth=3, n_estimators=100, random_state=42)
        model.fit(train[features], train['Target'])
        return model
    except Exception as e:
        logging.error(" Error in fit_arimax_model data: {}". format(e))

