import logging
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error

# # Function to predict and evaluate model 
def evaluate_model(dp):
    try:
        mae = mean_absolute_error(dp.price_actual, dp.price_predicted)
        print('ARIMA MAE:', mae)
    except Exception as e:
        logging.error(" Error in fit_arima_model data: {}". format(e))
        
def evaluate_arimax_model(dpx):
    try:
        mae = mean_absolute_error(dpx.price_actual, dpx.price_predicted)
        print('ARIMAX MAE:', mae)
    except Exception as e:
        logging.error(" Error in fit_arima_model data: {}". format(e))
        
def predict_model(train, test, features, model):
    try:
        model.fit(train[features], train['Target'])
        model_preds = model.predict(test[features])
        model_preds = pd.Series(model_preds, index=test.index, name='predictions')
        combine = pd.concat([test['Target'], model_preds], axis=1)
        return combine
    except Exception as e:
        logging.error(" Error in predict_model data: {}". format(e))
        
def backtest(data, model, features, start=5031, step=120):
    try:
        all_predictions = []
        for i in range(start, data.shape[0], step):
            train = data.iloc[:i].copy()
            test = data.iloc[i:(i+step)].copy()
            model_preds = predict_model(train, test, features, model)
            all_predictions.append(model_preds)
        return pd.concat(all_predictions)
    except Exception as e:
        logging.error(" Error in backtest data: {}". format(e))
        