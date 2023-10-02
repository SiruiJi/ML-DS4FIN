"""function for supervised regression model"""
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
"""function for data analysis and model evaluation"""
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
"""function for time series models"""
import statsmodels.api as sm
"""function for data import, preprocessing and visualization"""
import pandas_datareader.data as web
import datetime as dt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import warnings
warnings.filterwarnings('ignore')


def data_preparation(tsy_tickers, start_date, end_date):
    tsy_data = web.DataReader(tsy_tickers, 'fred', start_date, end_date).dropna(how='all').ffill()
    tsy_data['FDHBFIN'] = tsy_data['FDHBFIN'] * 1000
    tsy_data['GOV_PCT'] = tsy_data['TREAST'] / tsy_data['GFDEBTN']
    tsy_data['HOM_PCT'] = tsy_data['FYGFDPUN'] / tsy_data['GFDEBTN']
    tsy_data['FOR_PCT'] = tsy_data['FDHBFIN'] / tsy_data['GFDEBTN']

    Y = tsy_data.loc[:, ['DGS1MO', 'DGS2', 'DGS10']].shift(-5)
    Y.columns = Y.columns + '_pred'
    X = tsy_data.loc[:,
        ['DGS1MO', 'DGS3MO', 'DGS1', 'DGS2', 'DGS5', 'DGS7', 'DGS10', 'DGS30', 'TREAST', 'FYGFDPUN', 'FDHBFIN',
         'GFDEBTN', 'BAA10Y', 'GOV_PCT', 'HOM_PCT', 'FOR_PCT']]
    dataset = pd.concat([Y, X], axis=1).dropna().iloc[::5, :]
    X = dataset.loc[:, X.columns]
    Y = dataset.loc[:, Y.columns]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

    return dataset, X, Y,  X_train, X_test, Y_train, Y_test


def data_visualization(dataset, X, Y):
    Y.plot(style=['-', '--', ':'])
    plt.show()

    correlation = dataset.corr()
    plt.figure(figsize=(15, 15))
    plt.title('Correlation Matrix')
    sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
    plt.show()

    scatter_matrix(dataset, figsize=(15, 16))
    plt.show()

    X.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1, figsize=(12, 12))
    plt.show()

    X.plot(kind='density', subplots=True, layout=(5, 4), sharex=False, legend=True, fontsize=1, figsize=(15, 15))
    plt.show()

    X.plot(kind='box', subplots=True, layout=(5, 4), sharex=False, sharey=False, figsize=(15, 15))
    plt.show()

    for i in Y.columns:
        temp_Y = dataset[i]
        res = sm.tsa.seasonal_decompose(temp_Y, period=52)
        fig = res.plot()
        fig.set_figheight(8)
        fig.set_figwidth(15)
        plt.show()

    bestfeatures = SelectKBest(k=5, score_func=f_regression)
    for col in Y.columns:
        temp_Y = dataset[col]
        temp_X = dataset.loc[:, X.columns]
        fit = bestfeatures.fit(temp_X, temp_Y)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(X.columns)
        # concat two dataframes for better visualization
        featureScores = pd.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
        print(col)
        print(featureScores.nlargest(10, 'Score'))  # print 10 best features
        print('--------------')


def model_evaluation(X_train, X_test, Y_train, Y_test):
    # test options for regression
    num_folds = 10
    scoring = 'neg_mean_squared_error'

    # spot check the algorithms
    models = []
    models.append(('LR', LinearRegression()))
    models.append(('LASSO', Lasso()))
    models.append(('EN', ElasticNet()))
    models.append(('KNN', KNeighborsRegressor()))
    models.append(('CART', DecisionTreeRegressor()))
    # Neural Network
    models.append(('MLP', MLPRegressor()))

    trained_models = {}
    kfold_results = []
    names = []
    validation_results = []
    train_results = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds)
        # converted mean square error to positive. The lower the beter
        cv_results = -1 * cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        kfold_results.append(cv_results)
        names.append(name)


        # Finally we Train on the full period and test against validation
        res = model.fit(X_train, Y_train)
        validation_result = np.mean(np.square(res.predict(X_test) - Y_test))
        validation_results.append(validation_result)
        train_result = np.mean(np.square(res.predict(X_train) - Y_train))
        train_results.append(train_result)

        msg = "%s: \nAverage CV error: %s \nStd CV Error: (%s) \nTraining Error:\n%s \nTest Error:\n%s" % \
              (name, str(cv_results.mean()), str(cv_results.std()), str(train_result), str(validation_result))
        print(msg)
        print('----------')

    # compare algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(kfold_results)
    ax.set_xticklabels(names)
    fig.set_size_inches(15, 8)
    plt.show()

    # compare algorithms
    fig = plt.figure()

    ind = np.arange(len(names))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.bar(ind - width / 2, [x.mean() for x in train_results], width=width, label='Train Error')
    plt.bar(ind + width / 2, [x.mean() for x in validation_results], width=width, label='Validation Error')
    fig.set_size_inches(15, 8)
    plt.legend()
    ax.set_xticks(ind)
    ax.set_xticklabels(names)
    plt.show()


def ANNModel_parameter_selection(X_train, Y_train):
    num_folds = 10
    scoring = 'neg_mean_squared_error'
    param_grid = {'hidden_layer_sizes': [(20,), (50,), (20, 20), (20, 30, 20)]}
    model = MLPRegressor()
    kfold = KFold(n_splits=num_folds)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X_train, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

def create_ANNModel(X_train, Y_train):
    ANNModel = MLPRegressor(hidden_layer_sizes=(20, 30, 20))
    ANNModel.fit(X_train, Y_train)
    return ANNModel

def model_performance(X, Y):
    train_size = int(len(X) * (1 - 0.2))
    X_train, X_validation = X[0:train_size], X[train_size:len(X)]
    Y_train, Y_validation = Y[0:train_size], Y[train_size:len(X)]

    modelMLP = MLPRegressor(hidden_layer_sizes=(20, 30, 20))
    modelOLS = LinearRegression()
    model_MLP = modelMLP.fit(X_train, Y_train)
    model_OLS = modelOLS.fit(X_train, Y_train)

    Y_predMLP = pd.DataFrame(model_MLP.predict(X_validation)/100000, index=Y_validation.index,
                             columns=Y_validation.columns)

    Y_predOLS = pd.DataFrame(model_OLS.predict(X_validation), index=Y_validation.index,
                             columns=Y_validation.columns)

    pd.DataFrame({'Actual : 1m': Y_validation.loc[:, 'DGS1MO_pred'],
                  'Prediction MLP 1m': Y_predMLP.loc[:, 'DGS1MO_pred'],
                  'Prediction OLS 1m': Y_predOLS.loc[:, 'DGS1MO_pred']}).plot(figsize=(10, 5))

    pd.DataFrame({'Actual : 5yr': Y_validation.loc[:, 'DGS2_pred'],
                  'Prediction MLP 5yr': Y_predMLP.loc[:, 'DGS2_pred'],
                  'Prediction OLS 5yr': Y_predOLS.loc[:, 'DGS2_pred']}).plot(figsize=(10, 5))

    pd.DataFrame({'Actual : 30yr': Y_validation.loc[:, 'DGS10_pred'],
                  'Prediction MLP 30yr': Y_predMLP.loc[:, 'DGS10_pred'],
                  'Prediction OLS 30yr': Y_predOLS.loc[:, 'DGS10_pred']}).plot(figsize=(10, 5))
    plt.show()


def next_day_prediction(X, Y):
    models = []
    models.append(('LR', LinearRegression()))
    models.append(('LASSO', Lasso()))
    models.append(('EN', ElasticNet()))
    models.append(('KNN', KNeighborsRegressor()))
    models.append(('CART', DecisionTreeRegressor()))

    trained_models = {}
    for name, model in models:
        trained_model = model.fit(X, Y)
        trained_models[name] = trained_model

    predictions = {}
    X_last = X_.iloc[-1].values.reshape(1, -1)
    for model_name, model in trained_models.items():
        prediction = model.predict(X_last)
        predictions[model_name] = prediction

    for model_name, prediction in predictions.items():
        print(f"Next day prediction by {model_name}: {prediction}")
    return trained_models, predictions


if __name__ == '__main__':
    start_date = dt.datetime(2015, 1, 1)
    end_date = dt.datetime(2023, 9, 30)
    tsy_tickers = ['DGS1MO', 'DGS3MO', 'DGS1', 'DGS2', 'DGS5', 'DGS7', 'DGS10', 'DGS30',
                   'TREAST',  # Treasury securities held by the federal reserve
                   'FYGFDPUN',  # Federal debt held by the public
                   'FDHBFIN',  # Federal debt held by international investors
                   'GFDEBTN',  # Federal debt: total public debt
                   'BAA10Y',  # Baa Corporate bond yield relative to yield on 10 year
                   ]
    dataset_, X_, Y_,  X_train_, X_test_, Y_train_, Y_test_ = data_preparation(tsy_tickers, start_date, end_date)
    data_visualization(dataset_, X_, Y_)
    model_evaluation(X_train_, X_test_, Y_train_, Y_test_)
    ANNModel_parameter_selection(X_train_, Y_train_)
    ANNmodel_ = create_ANNModel(X_train_, Y_train_)
    model_performance(X_, Y_)
    model_, prediction_ = next_day_prediction(X_, Y_)
