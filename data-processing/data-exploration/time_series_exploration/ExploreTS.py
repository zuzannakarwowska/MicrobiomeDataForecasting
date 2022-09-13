import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import seaborn as sns

### I: visualization

### II: TestStationarity

from statsmodels.tsa.stattools import adfuller 
from statsmodels.tsa.stattools import kpss


class TestStationarity:
    
    '''
    
    The following are the possible outcomes of applying both the tests

    Case 1: Both tests conclude that the given series is stationary – The series is stationary

    Case 2: Both tests conclude that the given series is non-stationary – The series is non-stationary

    Case 3: ADF concludes non-stationary and KPSS concludes stationary – The series is trend stationary. To make the series strictly stationary, the
    trend needs to be removed in this case. Then the detrended series is checked for stationarity.

    Case 4: ADF concludes stationary and KPSS concludes non-stationary – The series is difference stationary. Differencing is to be used to make series
    stationary. Then the differenced series is checked for stationarity.

    '''
        
    def calculate_adf(self, df, autolag = 't-stat'):
        
        
        ''' 
        
        Calculate using ADF test if series is trend-stationary 
        
        A time series has stationarity if a shift in time doesn’t 
        cause a change in the shape of the distribution; unit roots
        are one cause for non-stationarity.
        
        
        H0: series has unit root** (is not stationary)
        H1: series does not have unit root and is trend-stationary
        
        
        ** unit root: A unit root (also called a unit root process
        or a difference stationary process) is a stochastic trend
        in a time series, sometimes called a “random walk with drift”;
        If a time series has a unit root, it shows a systematic pattern
        that is unpredictable.
        
        Params 
        ---------
        df - dataframe with shape [timestep, features]
        regression_param:
        
        autolag: how many lags to use
        "t-stat" based choice of maxlag.  Starts with maxlag and drops a
        lag until the t-statistic on the last lag length is significant
        using a 5%-sized test.
        
        Returns
        ---------
        ADF_STATIONARITY_DF - dataframe with ADF results for
        each feature
        
        
        '''
    
        adf_pval = []
        stationarity = []
        
        for col in df.columns:
            
            x = df[col].values
            result = adfuller(x, autolag = autolag) 
            pvalue = result[1]
            adf_pval.append(pvalue)

            if pvalue < 0.05:
                stationarity.append('stationary')
                
            elif pvalue >= 0.05:
                stationarity.append('non-stationary')

        ADF_STATIONARITY_DF = pd.DataFrame(list(zip(adf_pval, df.columns, stationarity)), 
                                           columns = ['adf_pval', 'feature', 'adf_stationarity'])

        return ADF_STATIONARITY_DF

    def calculate_kpss(self, df, regression = 'ct'):

        '''

        KPSS test tests if a time series is stationary around
        a mean or linear trend, or is non-stationary due to a
        unit root (random walk with drift).

        H0: data is trend-stationary
        H1: data is not stationary has unit root

        The KPSS test is based on linear regression. It breaks
        up a series into three parts: a deterministic trend (βt),
        a random walk (rt), and a stationary error (εt).
        If the data is stationary, it will have a fixed element
        for an intercept or the series will be stationary around
        a fixed level.

        ** data should be log transformed before this test 
        to turn any exponential trends into linear ones.
        
        Params 
        ---------
        df - dataframe with shape [timestep, features]
        regression_param:
        
        "c" : The data is stationary around a constant 
        "ct" : The data is stationary around a trend (default)
        
        Returns
        ---------
        KPSS_STATIONARITY_DF - dataframe with KPSS results for
        each feature

        '''

        kpss_pval = []
        stationarity = []

        for col in df.columns:

            x = df[col].values
            result = kpss(x, regression=regression) 
            pvalue = result[1]
            kpss_pval.append(pvalue)

            if pvalue >= 0.05:
                stationarity.append('stationary')

            elif pvalue < 0.05:
                stationarity.append('non-stationary')

        KPSS_STATIONARITY_DF = pd.DataFrame(list(zip(kpss_pval, df.columns, stationarity)),
                                            columns = ['kpss_pval', 'feature', 'kpss_stationarity'])

        return KPSS_STATIONARITY_DF

    
    
### III: autocorrelation

from statsmodels.tsa.stattools import acf, pacf

class AutoCorrelation:
    
    
    def __init__(self, lags=5):
        
        self.lags = lags
        
    def calculate_acf(self, df):

        ''' 
        
        calculate autocorrelation for each feature in given df

        params
        ---------
        df - dataframe with features as columns and rows as samples
        lags - max lag for autocorrelation calculation

        returns
        ---------
        df with pvalue, autocorrelation coeff, lag and feature name

        '''

        ACF_DF = []

        for col in df.columns:

            idx = [i for i in range(1, self.lags+1)]

            acf_results = acf(df[col],
                              nlags=self.lags,
                              adjusted=True,
                              qstat = True
                             )

            pvalue = acf_results[2]
            coeff = acf_results[0][1:]

            acf_df = pd.DataFrame(list(zip(coeff, pvalue, idx)),
                                  columns = ['coeff', 'pvalues', 'lag'])
            acf_df['feature'] = col
            ACF_DF.append(acf_df)


        ACF_DF = pd.concat(ACF_DF)

        return ACF_DF
    
    def calculate_pacf(self, df):
    
        ''' 
        
        calculate partial autocorrelation for each feature in given df

        params
        ---------
        df - dataframe with features as columns and rows as samples
        lags - max lag for partial autocorrelation calculation

        returns
        ---------
        df with partial autocorrelation coeff, lag and feature name

        '''
    
        lags = self.lags + 1 #pacf starts from lag 0 which is always perfectly correlated 
        
        PACF_DF = []
        
        for col in df.columns:
            
            pacf_results = pacf(df[col].values,
                                nlags=lags,
                                method='ols',
                                alpha = .05)
            
            coeff = pacf_results[0]
            
            idx = [i for i in range(0, lags)]
            pacf_df = pd.DataFrame(list(zip(coeff, idx)),
                                   columns = ['coeff', 'lags']) 
            pacf_df['feature'] = col
            
            PACF_DF.append(pacf_df)
            
        PACF_DF = pd.concat(PACF_DF)
        PACF_DF = PACF_DF[PACF_DF['lags'] > 0]
        
        return PACF_DF
    
    
### IV: Granger causality 

from statsmodels.tsa.stattools import grangercausalitytests

class GrengerCausality:
    
    def test_granger(self, df, maxlag=2):

        '''

        The Granger causality test is a statistical hypothesis
        test for determining whether one time series is useful
        in forecasting another.

        '''

        GRANGER_DF = []

        for i in range(len(df.columns)):
            for j in range(len(df.columns)):

                asv1 = df.iloc[:, i]
                asv2 = df.iloc[:, j]

                X = np.stack((asv1, asv2), axis=1)

                maxlag = maxlag
                test_result = grangercausalitytests(X, maxlag=maxlag, verbose = False)

                test = 'ssr_chi2test'
                pvalues = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
                f_size = [round(test_result[i+1][0][test][0],4) for i in range(maxlag)]

                granger_causality_df = pd.DataFrame(list(zip(f_size, pvalues)),
                                                    columns = ['effect_size', 'pvalues'])
                granger_causality_df['asv1'] = asv1.name
                granger_causality_df['asv2'] = asv2.name
                granger_causality_df['lag'] = [i+1 for i in range(maxlag)]

                GRANGER_DF.append(granger_causality_df)
                
        GRANGER_DF = pd.concat(GRANGER_DF)

        return GRANGER_DF

    def to_matrix(self, granger_df, value='pvalues', lag=1):

        '''

        convert results from test_granger() into an n x n matrix

        Params
        -------
        granger_df - df with granger test results from test_granger()
        lag - results from which lag to use in matrix (default 1)
        value - wether to use pvalues or effect size as matrix cells
        
        if value is set to pvalues
        if value is set to effect_size

        Returns
        -------
        granger_matrix - matrix of size n x n with pvalues or effect size
        of granger test

        '''

        if value == 'pvalues':

            granger_df = granger_df.sort_values(by = ['lag'])
            lag_df = granger_df[granger_df['lag'] == lag]
            lag_df = lag_df.sort_values(by = ['asv1', 'asv2'])

            granger_matrix = lag_df.pivot_table(index = 'asv1', columns = 'asv2', values = 'pvalues')

        elif value == 'effect_size':

            #filter significant effect sizes by replacing non-significant ones with 0
            granger_df = granger_df.loc[granger_df.pvalues > 0.05, ['effect_size']] = 0

            granger_df = granger_df.sort_values(by = ['lag'])
            lag_df = granger_df[granger_df['lag'] == lag]
            lag_df = lag_df.sort_values(by = ['asv1', 'asv2'])

            granger_matrix = lag_df.pivot_table(index = 'asv1', columns = 'asv2', values = 'effect_size')

        return granger_matrix
    
    def plot_heatmap(self, granger_matrix):
        
        title='Granger causality heatmap'
        
        plt.figure(figsize=(5, 5))

        sns.heatmap(granger_matrix.values,
                    cmap = 'mako_r')

        plt.title(title)
        plt.show()
    
    def plot_clustermap(self, granger_matrix):

        sns.clustermap(granger_matrix.values,
                       cmap="Blues",
                       dendrogram_ratio=(.1, .1),
                       figsize=(5, 5),
                       alpha = .7,
                       yticklabels=False,
                       xticklabels=False
                      )
        plt.show()
        
        
### V. Make Series Stationary

class MakeSeriesStationary:
    
    def remove_trend(self, df):
        
        ''' 
        run simple ols regression between series and timestep
        to detect trend and then remove it
        
        '''
        
        DETRENDED_DF = pd.DataFrame()
        
        for col in df.columns:

            series = df[col]
            y = series.values
            X = series.index.values
            X = sm.add_constant(X)

            ols_model = sm.OLS(y, X).fit()
            trend = ols_model.predict(X)

            y_detrended = y - trend
            DETRENDED_DF[series.name] = y_detrended
        
        return DETRENDED_DF

    def difference(self, df, period = 1):

        '''
        difference all columns in dataframe

        Params
        ----------
        df - dataframe of shape
        period - periods to shift for calculating difference, accepts negative
        values.

        Returns
        ----------
        diff_df - differences dataframe 
        '''
        diff_df = pd.DataFrame()

        for col in df.columns:

            diff_col = df[col].diff(period)
            diff_df[col] = diff_col

        return diff_df
