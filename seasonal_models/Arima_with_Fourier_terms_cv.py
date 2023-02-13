import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.fft import fft, fftfreq, ifft, rfft, rfftfreq
from scipy import signal as sig
from cmath import phase

data = pd.read_csv('/storage/zkarwowska/microbiome-dynamics-preprint/datasets/alpha_diversity/male_shannon.csv')
data.iloc[200:240] = data.iloc[200:240].rolling(10, min_periods=1).mean()

def make_fft(amp, N):
    
    ''' prepare fourier series of amplitude amp and length N '''

    N = N 
    T = 1/1 
    f = 1/amp

    x = np.linspace(0, N*T, N, endpoint=False)
    ft = np.cos(f * 2.0*np.pi*x) + np.sin(f * 2.0*np.pi*x)

    return ft

def sliding_arima_with_fft(data, exog_fft, p, d, q):
    
    data = data
    exog_fft = exog_fft
    p=p
    d=d
    q=q
    
    train_fold_size = 50; test_fold_size = 25

    train_rho = []; test_rho = []
    for n in range(0, 350, 20):

        # define train and test
        train = data[n:n+train_fold_size].values
        test = data[n+train_fold_size:(n+train_fold_size+test_fold_size)].values

        #create exog fft variable
        train_fft = exog_fft[n:n+train_fold_size]
        test_fft = exog_fft[n+train_fold_size:(n+train_fold_size+test_fold_size)]

        # train ARIMA model

        arima_model = ARIMA((train), order = (p, d, q), exog=train_fft)
        arima_model.initialize_approximate_diffuse() 
        arima_model_fit = arima_model.fit(method_kwargs={"warn_convergence": False})

        #predict train
        yhat_train = arima_model_fit.predict(exog=train_fft)
        yhat_test = arima_model_fit.predict(start=train_fold_size, end=((train_fold_size+test_fold_size)-1), exog=test_fft)
        
        train_score = stats.spearmanr(train, yhat_train)[0]
        test_score = stats.spearmanr(test, yhat_test)[0]

        train_rho.append(train_score)
        test_rho.append(test_score)

    df = pd.DataFrame(list(zip(train_rho, test_rho)), columns = ['train_score', 'test_score'])
    df[['p', 'q', 'd']] = p, q, d
    df['folds'] = range(0, 18)
    
        
    return df


FFTS = [i for i in range(2, 30)]
Ps = range(0, 17)
Qs = range(0, 17)
Ds = [0, 1]

DF = []
for f in FFTS:
    for p in Ps:
        for q in Qs:
            for d in Ds:
                exog_fft = make_fft(f, len(data))
                df = sliding_arima_with_fft(data, exog_fft, p, d, q)
                df['freq'] = f
                DF.append(df)
                
arima_fft_results_df = pd.concat(DF)

arima_fft_results_df.to_csv('male_arima_fft_cv.csv')