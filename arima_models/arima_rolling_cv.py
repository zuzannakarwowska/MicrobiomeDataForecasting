import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import arch

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy import stats
import warnings
import random 

warnings.filterwarnings('ignore')


shannon_df = pd.read_csv('/storage/zkarwowska/microbiome-dynamics-preprint/datasets/alpha_diversity/male_shannon.csv')

validation_set = shannon_df.iloc[-43:] ## leave last 43 days for validation
train_set = shannon_df.iloc[:-43] ## train set used for n fold corss validation

def arima_rolling_forecast(train, p, q, test):

    history = [x for x in train]

    predictions = list()
    for t in range(len(test)):
        
        history_log = np.log(history)
        arima_model = ARIMA(history_log, order=(p, 0, q))
        arima_model.initialize_approximate_diffuse() 
        arima_model_fit = arima_model.fit()

        yhat = arima_model_fit.forecast(nsteps=1)[0]
        predictions.append(yhat)
        history.append(test[t])
        
    return np.exp(predictions)


def arima_cv(X,
             n_folds, 
             fold_size,
             p,
             q):
    
    data = X.values.reshape(len(X), )

    n_folds = n_folds
    fold_size = fold_size


    test_fold_size = int(fold_size * 0.2)
    train_fold_size = fold_size - test_fold_size

    init_n = random.sample(range(0, 200), n_folds)

    SPEARMAN = []
    MAPE = []

    for n in init_n:

        train = data[n:n+train_fold_size]
        test = data[n+train_fold_size:(n+train_fold_size+test_fold_size)]

        yhat = arima_rolling_forecast(train, p, q, test)
        spearman_rho = stats.spearmanr(test, yhat)[0]
        mape = mean_absolute_percentage_error(test, yhat)

        SPEARMAN.append(spearman_rho)
        MAPE.append(mape)

    results_df = pd.DataFrame(list(zip(SPEARMAN, MAPE)), columns = ['rho', 'mape'])
    results_df['fold'] = [i for i in range(0, n_folds)]
    results_df[['p', 'q']] = p, q
    
    return results_df


Ps = range(2, 20)
Qs = range(0, 5)

DF = []
for p in Ps:
    for q in Qs:
        
        df = arima_cv(train_set, 5, 200, p, q)
        df.to_csv('arima_scores{}_{}.csv'.format(p, q))


        DF.append(df)
        
DF = pd.concat(DF)
