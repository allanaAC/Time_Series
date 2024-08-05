import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, precision_score
from xgboost import XGBClassifier
import yfinance as yf

def plot_univariate(df):
    try:
        plot = sns.lineplot(df['AAPL'])
        plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
        plt.show()
    except Exception as e:
        logging.error(" Error in preprocess_data data: {}". format(e))
        
def decompose_time_series(df):
    try:
        decomposed = seasonal_decompose(df['AAPL'])
        trend = decomposed.trend
        seasonal = decomposed.seasonal
        residual = decomposed.resid
        plt.figure(figsize=(12,8))
        plt.subplot(411)
        plt.plot(df['AAPL'], label='Original', color='black')
        plt.legend(loc='upper left')
        plt.subplot(412)
        plt.plot(trend, label='Trend', color='red')
        plt.legend(loc='upper left')
        plt.subplot(413)
        plt.plot(seasonal, label='Seasonal', color='blue')
        plt.legend(loc='upper left')
        plt.subplot(414)
        plt.plot(residual, label='Residual', color='black')
        plt.legend(loc='upper left')
        plt.show()
    except Exception as e:
        logging.error(" Error in decompose_time_series data: {}". format(e))
        
def test_stationarity(df):
    try:
        results = adfuller(df['AAPL'])
        print('ADF p-value:', results[1])
        v1 = df['AAPL'].diff().dropna()
        results1 = adfuller(v1)
        print('Differenced ADF p-value:', results1[1])
        plt.plot(v1)
        plt.title('1st Order Differenced Series')
        plt.xlabel('Date')
        plt.xticks(rotation=30)
        plt.ylabel('Price (USD)')
        plt.show()
        return v1
    except Exception as e:
        logging.error(" Error in test_stationarity data: {}". format(e))
        
def plot_acf_pacf(df):
    try:
        plt.rcParams.update({'figure.figsize': (7, 4), 'figure.dpi': 80})
        plot_acf(df['AAPL'].dropna())
        plt.show()
        plot_pacf(df['AAPL'].dropna(), lags=11)
        plt.show()
    except Exception as e:
        logging.error(" Error in plot_acf_pacf data: {}". format(e))
        
def plot_forecast(data, forecast):
    try:
        ypred = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=0.05)
        Date = pd.Series(['2024-01-01', '2024-02-01'])
        price_actual = pd.Series(['184.40', '185.04'])
        price_predicted = pd.Series(ypred.values)
        lower_int = pd.Series(conf_int['lower AAPL'].values)
        upper_int = pd.Series(conf_int['upper AAPL'].values)
        dp = pd.DataFrame([Date, price_actual, lower_int, price_predicted, upper_int], index=['Date', 'price_actual', 'lower_int', 'price_predicted', 'upper_int']).T
        dp = dp.set_index('Date')
        dp.index = pd.to_datetime(dp.index)
        plt.plot(data.AAPL)
        plt.plot(dp.price_predicted, color='orange')
        plt.fill_between(dp.index, lower_int, upper_int, color='k', alpha=.15)
        plt.title('Model Performance')
        plt.legend(['Actual', 'Prediction'], loc='lower right')
        plt.xlabel('Date')
        plt.xticks(rotation=30)
        plt.ylabel('Price (USD)')
        plt.show()
        return dp
    except Exception as e:
        logging.error(" Error in plot_forecast data: {}". format(e))
        
def plot_arimax_forecast(data, forecast):
    try:
        ypred = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=0.05)
        Date = pd.Series(['2024-01-01', '2024-02-01'])
        price_actual = pd.Series(['184.40', '185.04'])
        price_predicted = pd.Series(ypred.values)
        lower_int = pd.Series(conf_int['lower AAPL'].values)
        upper_int = pd.Series(conf_int['upper AAPL'].values)
        dpx = pd.DataFrame([Date, price_actual, lower_int, price_predicted, upper_int], index=['Date', 'price_actual', 'lower_int', 'price_predicted', 'upper_int']).T
        dpx = dpx.set_index('Date')
        dpx.index = pd.to_datetime(dpx.index)
        plt.plot(data.AAPL)
        plt.plot(dpx.price_predicted, color='orange')
        plt.fill_between(dpx.index, lower_int, upper_int, color='k', alpha=.15)
        plt.title('ARIMAX Model Performance')
        plt.legend(['Actual', 'Prediction'], loc='lower right')
        plt.xlabel('Date')
        plt.xticks(rotation=30)
        plt.ylabel('Price (USD)')
        plt.show()
        return dpx
    except Exception as e:
        logging.error(" Error in plot_arimax_forecast data: {}". format(e))
        
def evaluate_xgboost_model(model, test, features):
    try:
        model_preds = model.predict(test[features])
        model_preds = pd.Series(model_preds, index=test.index)
        precision = precision_score(test['Target'], model_preds)
        plt.plot(test['Target'], label='Actual')
        plt.plot(model_preds, label='Predicted')
        plt.legend()
        plt.show()
        return precision
    except Exception as e:
        logging.error(" Error in evaluate_xgboost_model data: {}". format(e))
    
   