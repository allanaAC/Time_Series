import warnings
from sklearn.metrics import precision_score
from src.data.load_dataset import load_and_preprocess_data, preprocess_data, load_yfinance_data
from src.visualization.visualize import plot_arimax_forecast, evaluate_xgboost_model
from src.visualization.visualize import plot_univariate, decompose_time_series, test_stationarity, plot_acf_pacf, plot_forecast
from src.feature.build_features import create_features
from src.model.train_model import fit_arima_model, fit_arimax_model, train_xgboost_model
from src.model.predict_model import evaluate_model, evaluate_arimax_model, backtest


if __name__ == "__main__":
    # Suppress warnings
    warnings.filterwarnings('ignore')
    # Load and preprocess the data
     # For univariate analysis
    data = load_and_preprocess_data('src/data/AAPL.csv')
    df = preprocess_data(data)
    plot_univariate(df)
    decompose_time_series(df)
    v1 = test_stationarity(df)
    plot_acf_pacf(df)
    forecast = fit_arima_model(df)
    dp = plot_forecast(data, forecast)
    evaluate_model(dp)

    # For ARIMAX analysis
    """ dfx = data.iloc[0:-2, 0:3]
    forecast_arimax = fit_arimax_model(dfx)
    dpx = plot_arimax_forecast(data, forecast_arimax)
    evaluate_arimax_model(dpx) """
    
     # For XGBoost analysis
    data = load_yfinance_data("AAPL", "2000-01-01", "2022-05-31")
    data = create_features(data)
    train = data.iloc[:-30]
    test = data.iloc[-30:]
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    model = train_xgboost_model(train, features)
    precision = evaluate_xgboost_model(model, test, features)
    print('XGBoost Precision Score:', precision)

    # Backtesting
    predictions = backtest(data, model, features)
    print('Backtesting Precision Score:', precision_score(predictions['Target'], predictions['predictions']))
