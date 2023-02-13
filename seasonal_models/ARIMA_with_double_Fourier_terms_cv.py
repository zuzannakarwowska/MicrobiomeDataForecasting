import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.fft import fft, fftfreq, ifft, rfft, rfftfreq
from scipy import signal as sig
from cmath import phase


sns.set_style('whitegrid')

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100
warnings.filterwarnings("ignore")

data = pd.read_csv('/storage/zkarwowska/microbiome-dynamics-preprint/datasets/alpha_diversity/male_shannon.csv')

def make_fft(freq, N):
    
    ''' prepare fourier series of frequency amp and length N '''

    N = N 
    T = 1/1 
    f = 1/freq

    x = np.linspace(0, N*T, N, endpoint=False)
    ft = np.cos(f * 2.0*np.pi*x) + np.sin(f * 2.0*np.pi*x)

    return ft


def sliding_arima_with_fft(data, exog_fft, p, d, q):
    
    data = data
    exog_fft = exog_fft
    p=p
    d=d
    q=q
    
    m = data.mean().values[0]
    train_fold_size = 50; test_fold_size = 30

    train_rho = []; test_rho = []
    for n in range(0, 350, 20):

        # define train and test
        train = data[n:n+train_fold_size].values
        test = data[n+train_fold_size:(n+train_fold_size+test_fold_size)].values

        # detrend 
        train = train - m

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
        yhat_test = yhat_test + m
        plt.plot(test, 'k')
        plt.plot((yhat_test), 'r')
        plt.show()
        
        train_score = stats.spearmanr(train, yhat_train)[0]
        test_score = stats.spearmanr(test, yhat_test)[0]

        train_rho.append(train_score)
        test_rho.append(test_score)

    df = pd.DataFrame(list(zip(train_rho, test_rho)), columns = ['train_score', 'test_score'])
    df[['p', 'q', 'd']] = p, q, d
    df['folds'] = range(0, len(range(0, 350, 20)))
    
        
    return df

SEASONALITIES = range(2, 40, 1)
ps = range(0, 15)
qs = range(0, 15)
ds = [0, 1]

DF = []
for f1 in SEASONALITIES:
    for f2 in SEASONALITIES:
        for p in ps:
            for d in ds:
                for q in qs:                    
                    
                    exog_fft = make_fft(f1, len(data)) + make_fft(f2, len(data))
                    df = sliding_arima_with_fft(data, exog_fft, p, d, q)
                    df[['f1', 'f2']] = f1, f2
                    DF.append(df)
results_df = pd.concat(DF)
results_df.to_csv('male_fft_arima_cv_results.csv')