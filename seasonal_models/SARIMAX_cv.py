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

def test_sarimax(data, p, d, q, P, D, Q, S):
    
    p=p; d=d; q=q
    P =P; Q=Q; D=D; S = S
    m = data.mean().values[0]
    
    test_size = 20; train_size = 50

    train_rho = []; test_rho = []
    for n in range(0, 300, 25):

        train = data.iloc[n:(n+train_size)]
        test = data.iloc[(n+train_size):(n+train_size+test_size)]

        train = train - m

        #arima_model = ARIMA(train, order = (p, d, q))
        arima_model = SARIMAX(train,  order = (p, d, q), seasonal_order=(P, D, Q, S))
        arima_model.initialize_approximate_diffuse() 
        arima_model_fit = arima_model.fit(method_kwargs={"warn_convergence": False}, disp=False)

        #predict train
        yhat_train = arima_model_fit.predict()
        yhat_test = arima_model_fit.predict(start=train_size, end = (train_size + test_size)-1)
        yhat_test = yhat_test + m

        train_score = stats.spearmanr(train, yhat_train)[0]
        test_score = stats.spearmanr(test, yhat_test)[0]

        train_rho.append(train_score)
        test_rho.append(test_score)

    df = pd.DataFrame(list(zip(train_rho, test_rho)), columns = ['train_rho', 'test_rho'])

    df['fold'] = range(0, len(range(0, 300, 25)))
    df[['p', 'd', 'q']] = p, d, q
    df[['P', 'D', 'Q', 'S']] = P, D, Q, S
    
    return df

data = pd.read_csv('/storage/zkarwowska/microbiome-dynamics-preprint/datasets/alpha_diversity/male_shannon.csv')
data.iloc[200:240] = data.iloc[200:240].rolling(10, min_periods=1).mean()
subject = 'male'

ps = range(0, 13)
qs = range(0, 13)
ds = [0, 1]

Ps = range(0, 5)
Qs = range(0, 5)
Ds = [0, 1]
S = 20

DF = []
for p in ps:
    for q in qs:
        for d in ds:
            for P in Ps:
                for Q in Qs:
                    for D in Ds:
                        
                        df = test_sarimax(data, p, d, q, P, D, Q, S)
                        DF.append(df)
                        
results_df = pd.concat(DF)
results_df.to_csv('male_sarimax_cv.csv')