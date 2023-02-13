import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.preprocessing import FourierFeaturizer
import arch

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, roc_auc_score, f1_score

from scipy import stats
import warnings
import random 

warnings.filterwarnings('ignore')

sns.set_style('whitegrid')


shannon_df = pd.read_csv('/storage/zkarwowska/microbiome-dynamics-preprint/datasets/alpha_diversity/male_shannon.csv')

validation_set = shannon_df.iloc[-43:] ## leave last 43 days for validation
train_set = shannon_df.iloc[:-43] ## train set used for n fold corss validation

def find_outliers(y):

        y_series = pd.DataFrame(y, columns = ['y']) 
        rolling_mean = y_series.ewm(span=7, adjust=False).mean()
        rolling_std = y_series.ewm(span=7, adjust=False).std()

        std_pos = rolling_mean + rolling_std
        std_neg = rolling_mean - rolling_std

        conditions = [
            (y_series > std_pos),
            (y_series <  std_neg),
            (y_series < std_pos) & (y_series >  std_neg)]

        choices = [1, -1, 0]
        y_series['outlier'] = np.select(conditions, choices, default=0)

        return y_series
    
def calculate_outlier_score(y, yhat):
    
    y_test_outliers = find_outliers(y)
    y_hat_outliers = find_outliers(yhat)

    score1 = np.round(f1_score(y_test_outliers.outlier, y_hat_outliers.outlier, average = 'macro'), 3)

    return score1

def arimax_rolling_forecast(train, exog_train, p, q, test, exog_test):

    history = [x for x in train]
    exog_history = [x for x in exog_train]

    predictions = list()
    for t in range(len(test)):

        history_log = np.log(history)
        sarima_model = ARIMA(history_log, order = (p, 0, q), exog = exog_history)
        sarima_model_fit = sarima_model.fit()

        yhat = sarima_model_fit.forecast(nsteps=1, exog = exog_test[t])[0]
        predictions.append(yhat)

        exog_history.append(exog_test[t])
        history.append(test[t])
        
    return np.exp(predictions)


def arimax_cv(X,
              f_terms,
             n_folds, 
             fold_size,
             p,
             q):
    
    data = X.values.reshape(len(X), )
    _ , fourier_term = FourierFeaturizer(m=f_terms, k=1).fit_transform(np.arange(len(X)))


    n_folds = n_folds
    fold_size = fold_size


    test_fold_size = int(fold_size * 0.3)
    train_fold_size = fold_size - test_fold_size


    init_n = random.sample(range(0, 200), n_folds)

    SPEARMAN = []
    MAPE = []
    F1 = []
    for n in init_n:

        train = data[n:n+train_fold_size]
        test = data[n+train_fold_size:(n+train_fold_size+test_fold_size)]
        
        exog_train = fourier_term[n:n+train_fold_size].values
        exog_test = fourier_term[n+train_fold_size:(n+train_fold_size+test_fold_size)].values
        
        yhat = arimax_rolling_forecast(train, exog_train, p, q, test, exog_test)
        

        spearman_rho = stats.spearmanr(test, yhat)[0]
        mape = mean_absolute_percentage_error(test, yhat)
        outlier_score = calculate_outlier_score(test, yhat)
        F1.append(outlier_score)
        
        SPEARMAN.append(spearman_rho)
        MAPE.append(mape)

    results_df = pd.DataFrame(list(zip(SPEARMAN, MAPE, F1)), columns = ['rho', 'mape', 'f1'])
    results_df['fold'] = [i for i in range(0, n_folds)]
    results_df[['p', 'q']] = p, q
    
    
    return results_df


for f in range(5, 30):
    for p in range (10, 20):
        for q in range(10, 20):
            
    
            res = arimax_cv(train_set, f, 5, 200, p, q)
            res.to_csv('arima_fft_{}_p{}_q{}_results.csv'.format(f, p, q))