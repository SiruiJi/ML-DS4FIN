"""function for supervised regression model"""
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
"""function for data analysis and model evaluation"""
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

"""function for deep learning models"""
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers import LSTM
"""function for time series models"""
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
"""function for data import, preprocessing and visualization"""
import yfinance as yf
import datetime as dt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import warnings


def data_preprocessing(stk_tickers, ccy_tickers, idx_tickers, start_date, end_date):

    skt_data = yf.download(stk_tickers, start_date, end_date)['Adj Close']
    ccy_data = yf.download(ccy_tickers, start_date, end_date)['Adj Close']
    idx_data = yf.download(idx_tickers, start_date, end_date)['Adj Close']

    return_period = 5
    Y = np.log(skt_data.loc[:, 'MSFT']).diff(return_period).shift(-return_period)
    '''Use the .diff(return_period) method to compute the difference between the current value and the value return_period 
    days ago. Shift the resulting series by -return_period days using the .shift(-return_period) method. This essentially 
    aligns the current date with the future return.'''
    Y.name = 'MSFT_pred'

    X1 = np.log(skt_data.loc[:, ('IBM', 'GOOGL')]).diff(return_period)
    X2 = np.log(ccy_data).diff(return_period)
    X3 = np.log(idx_data).diff(return_period)
    X4 = pd.concat([np.log(skt_data.loc[:, 'MSFT']).diff(i)
                    for i in [return_period, return_period * 3, return_period * 6, return_period * 12]],
                   axis=1).dropna()
    X4.columns = ['MSFT_DT', 'MSFT_3DT', 'MSFT_6DT', 'MSFT_12DT']
    X = pd.concat([X1, X2, X3, X4], axis=1).dropna()

    # dataset_daily = pd.concat([X, Y], axis=1).dropna()
    dataset = pd.concat([Y, X], axis=1).dropna().iloc[::return_period, :]
    ''' .iloc[::return_period,:] to extract weekly data'''
    Y = dataset.loc[:, Y.name]
    X = dataset.loc[:, X.columns]
    '''in this project, we want to use the lagged five_day return of stocks, currencies and indices along with lagged 
    5-day, 15-day, 30-day and 60 day return of MSFT as independent variables. '''

    test_size = 0.2
    train_size = int(len(X) * (1 - test_size))
    X_train = X[:train_size]
    X_test = X[train_size:]
    Y_train = Y[:train_size]
    Y_test = Y[train_size:]

    return dataset, Y, X, X_train, X_test, Y_train, Y_test


def data_visualization(dataset, Y):
    correlation = dataset.corr()
    plt.figure(figsize=(15, 15))
    plt.title('Correlation Matrix')
    sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
    plt.show()

    scatter_matrix(dataset, figsize=(12, 12))
    plt.show()

    res = sm.tsa.seasonal_decompose(Y, period=52)
    '''52 moving average'''
    fig = res.plot()
    fig.set_figheight(8)
    fig.set_figwidth(15)
    plt.show()


def LMSTModel(X ,X_train, X_test, Y_train, Y_test):

    seq_len = 2
    Y_train_LSTM, Y_test_LSTM = np.array(Y_train)[seq_len - 1:], np.array(Y_test)
    X_train_LSTM = np.zeros((X_train.shape[0] + 1 - seq_len, seq_len, X_train.shape[1]))
    X_test_LSMT = np.zeros((X_test.shape[0], seq_len, X.shape[1]))
    for i in range(seq_len):
        if i < seq_len - 1:  # For the first few sequences, use the tail of X_train
            X_test_LSMT[:seq_len - i - 1, i, :] = np.array(X_train)[-seq_len + i + 1:, :]
            X_test_LSMT[seq_len - i - 1:, i, :] = np.array(X_test)[:X_test.shape[0] - seq_len + i + 1, :]
        else:  # For the rest, just use X_test
            X_test_LSMT[:, i, :] = np.array(X_test)[:X_test.shape[0] - seq_len + i + 1, :]

    def create_LSMTmodel():
        model = Sequential()
        model.add(LSTM(50, input_shape=(X_train_LSTM.shape[1], X_train_LSTM.shape[2])))
        model.add(Dense(1))
        optimizer = SGD(learning_rate=0.01, momentum=0)
        model.compile(loss='mse', optimizer='adam')
        return model

    LSTMModel = create_LSMTmodel()
    LSTMModel.fit = LSTMModel.fit(X_train_LSTM, Y_train_LSTM, validation_data=(X_test_LSMT, Y_test_LSTM), epochs=330,
                                  batch_size=72, verbose=0, shuffle=False)
    error_Train_LSTM = mean_squared_error(Y_train_LSTM, LSTMModel.predict(X_train_LSTM))
    predict_LSTM = LSTMModel.predict(X_test_LSMT)
    error_Test_LSTM = mean_squared_error(Y_test, predict_LSTM)
    return error_Train_LSTM, error_Test_LSTM, X_train_LSTM, Y_train_LSTM

def find_best_arimax_order(ts, exog, max_p=10, max_d=5, max_q=10):
    """
    Find the best ARIMAX order for a given time series using grid search.

    Parameters:
    - ts (pd.Series): The time series data (dependent variable).
    - exog (pd.DataFrame): Exogenous variables (independent variables).
    - max_p (int): Maximum value for p.
    - max_d (int): Maximum value for d.
    - max_q (int): Maximum value for q.

    Returns:
    - best_order (tuple): The best (p, d, q) order.
    - best_aic (float): The AIC value for the best order.
    """

    best_aic = float('inf')
    best_order = None

    # Suppress warnings
    warnings.filterwarnings("ignore")

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(ts, order=(p, d, q), exog=exog)
                    result = model.fit()
                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_order = (p, d, q)
                except:
                    continue

    warnings.resetwarnings()

    return best_order, best_aic


def model_evaluate(X, X_train, X_test, Y_train, Y_test, error_Train_LSTM, error_Test_LSTM):
    num_folds = 10
    scoring = 'neg_mean_squared_error'

    models = []
    models.append(('LR', LinearRegression()))
    models.append(('LASSO', Lasso()))
    models.append(('EN', ElasticNet()))
    models.append(('KNN', KNeighborsRegressor()))
    models.append(('CART', DecisionTreeRegressor()))
    models.append(('SVR', SVR()))

    models.append(('MLP', MLPRegressor()))

    models.append(('ABR', AdaBoostRegressor()))
    models.append(('GBR', GradientBoostingRegressor()))
    models.append(('RFR', RandomForestRegressor()))
    models.append(('ETR', ExtraTreesRegressor()))

    modelARIMA = ARIMA(endog=Y_train, exog=X_train, order=(1, 0, 1))
    modelARIMA_fit = modelARIMA.fit()
    error_Train_ARIMA = mean_squared_error(Y_train, modelARIMA_fit.fittedvalues)
    predict = modelARIMA_fit.predict(start=len(X_train) - 1, end=len(X) - 1, exog=X_test)[1:]
    error_test_ARIMA = mean_squared_error(Y_test, predict)

    names = []
    kfold_results = []
    test_results = []
    train_results = []
    for name, model in models:
        names.append(name)
        kfold = KFold(n_splits=num_folds)
        cv_results = -1 * cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        kfold_results.append(cv_results)
        regressor = model.fit(X_train, Y_train)
        train_result = mean_squared_error(regressor.predict(X_train), Y_train)
        train_results.append(train_result)
        test_result = mean_squared_error(regressor.predict(X_test), Y_test)
        test_results.append(test_result)


    fig = plt.figure()
    fig.suptitle('Algorithm Comparison: Kfold results')
    ax = fig.add_subplot(111)
    plt.boxplot(kfold_results)
    ax.set_xticklabels(names)
    fig.set_size_inches(15, 8)
    plt.show()

    names.append("ARIMA")
    names.append("LSTM")
    test_results.append(error_test_ARIMA)
    test_results.append(error_Test_LSTM)
    train_results.append(error_Train_ARIMA)
    train_results.append(error_Train_LSTM)
    fig = plt.figure()
    ind = np.arange(len(names))
    width = 0.35
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.bar(ind - width / 2, train_results, width=width, label='Train Error')
    plt.bar(ind + width / 2, test_results, width=width, label='Test Error')
    fig.set_size_inches(15, 8)
    plt.legend()
    ax.set_xticks(ind)
    ax.set_xticklabels(names)
    plt.show()


def model_performance(X_train, X_test, Y_train, Y_test):
    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)
    predict = pd.DataFrame(regressor.predict(X_test))
    predict.index = Y_test.index
    plt.plot(np.exp(Y_test).cumprod(), 'r', label='actual')
    plt.plot(np.exp(predict).cumprod(), 'b--', label='predicted')
    plt.legend()
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.show()


def next_day_prediction(X, Y, X_train_LSTM, Y_train_LSTM):
    # Split the data into training and testing sets
    test_size = 0.2
    train_size = int(len(X) * (1 - test_size))
    X_train = X[:train_size]
    X_test = X[train_size:]
    Y_train = Y[:train_size]
    Y_test = Y[train_size:]

    # Models to consider
    models = [
        ('LR', LinearRegression()),
        ('LASSO', Lasso()),
        ('EN', ElasticNet()),
        ('KNN', KNeighborsRegressor()),
        ('CART', DecisionTreeRegressor()),
        ('SVR', SVR()),
        ('MLP', MLPRegressor()),
        ('ABR', AdaBoostRegressor()),
        ('GBR', GradientBoostingRegressor()),
        ('RFR', RandomForestRegressor()),
        ('ETR', ExtraTreesRegressor())
    ]

    # Store predictions for each model
    predictions = {}

    # Iterate over each model, fit the model, and predict the next day's return
    for name, model in models:
        model.fit(X_train, Y_train)
        prediction = model.predict(X_test.iloc[-1].values.reshape(1, -1))
        predictions[name] = prediction[0]

    # Special handling for ARIMA model
    arima_order = (1, 0, 1)
    modelARIMA = ARIMA(endog=Y_train, exog=X_train, order=arima_order)
    modelARIMA_fit = modelARIMA.fit()
    n_steps = len(X_test)
    exog_data = np.tile(X_test.iloc[-1].values, (n_steps, 1))
    arima_prediction = modelARIMA_fit.predict(start=len(X_train) - 1, end=len(X) - 1, exog=exog_data)[1:]

    predictions["ARIMA"] = arima_prediction.iloc[0]

    def create_LSMTmodel():
        model = Sequential()
        model.add(LSTM(50, input_shape=(X_train_LSTM.shape[1], X_train_LSTM.shape[2])))
        model.add(Dense(1))
        optimizer = SGD(learning_rate=0.01, momentum=0)
        model.compile(loss='mse', optimizer='adam')
        return model

    lstm_model = create_LSMTmodel()  # Assuming create_LSMTmodel() is a separate function in your code
    lstm_model.fit(X_train_LSTM, Y_train_LSTM)
    last_values = X_test[-2:].values
    reshaped_values = last_values.reshape(1, X_train_LSTM.shape[1], X_train_LSTM.shape[2])

    lstm_prediction = lstm_model.predict(reshaped_values)
    predictions["LSTM"] = lstm_prediction[0][0]

    return predictions


if __name__ == '__main__':
    start_date = dt.datetime(2020, 1, 1)
    end_date = dt.datetime(2023, 9, 30)
    stk_tickers = ['MSFT', 'IBM', 'GOOGL']
    ccy_tickers = ['JPY=X', 'GBP=X']
    idx_tickers = ['^GSPC', '^DJI', '^VIX']
    dataset_, Y_, X_, X_train_, X_test_, Y_train_, Y_test_ = data_preprocessing(stk_tickers, ccy_tickers, idx_tickers, start_date, end_date)
    # data_visualization(dataset_, Y_)
    error_Train_LSTM, error_Test_LSTM,X_train_LSTM_, Y_train_LSTM_ = LMSTModel(X_, X_train_, X_test_, Y_train_, Y_test_)
    find_best_arimax_order()
    model_evaluate(X_, X_train_, X_test_, Y_train_, Y_test_, error_Train_LSTM, error_Test_LSTM)
    model_performance(X_train_, X_test_, Y_train_, Y_test_)
    prediction = next_day_prediction(X_, Y_, X_train_LSTM_, Y_train_LSTM_)
    print(prediction)
