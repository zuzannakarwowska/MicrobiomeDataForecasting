import pandas as pd
import numpy as np
import random
import os
import sys


from scipy.special import logsumexp
from statsmodels.tsa.api import VAR

MODULE_PATH = os.path.abspath('/storage/pszczerbiak/microbiome_interactions_project') 
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

from utils.transformers import CLRTransformer


class VAR_prediction:

    def train_test_split(self, data, split_prc):
        
        '''
        Define x_train and x_test.

        Params
        ------
        data      : np.array
        split_prc : how many rows to take as test set
        
        Returns
        ------
        x_train   : np.array
        x_test    : np.array
        
        '''
        
        split = int(data.shape[0] * split_prc) 
        x_train = data[:-split].values
        x_test = data[-split:].values
    
        return x_train, x_test
        
        
    def fit_var_model(self, x_train, lag=1):
        
        '''
        Fit VAR model to train data

        Params
        ------
        data: np.array 
        lag:  
        
        Returns
        ------
        model_fit: fitted model
        
        '''
        
        lag = lag

        model = VAR(x_train)
        model_fit = model.fit(lag)
        
        return model_fit
        
    def VAR_predict_train_shuffled(self, x_train, fitted_model, x_test , steps = 1):
        
        '''
        Use fitted VAR model to make predictions
        on train data in a shuffled manner (use random days,
        do not care about order)

        Params
        ------
        x_train: np.array
        x_test: np.array
        fitted_model: fitted VAR model
        steps: forecast horizon
        
        Returns
        ------
        predictions_df 
        
        '''
        
        predictions = []

        forecast_input = x_train[-1].reshape(steps, len(x_train[-1])) 
        yhat = fitted_model.forecast(y=forecast_input, steps=steps)
        predictions.append(yhat)

        random_idx = list(range(len(x_test) - 1)) 
        random.Random(10).shuffle(random_idx)

        for i in random_idx:
            forecast_input = x_test[i].reshape(steps, len(x_test[i]))
            yhat = fitted_model.forecast(y=forecast_input, steps=steps)
            predictions.append(yhat)

        predictions_array = [item for sublist in predictions for item in sublist]
        predictions_df = pd.DataFrame(predictions_array)
        
        idx = [0] + [x+1 for x in random_idx]
        predictions_df.index = idx 
        predictions_df = predictions_df.sort_index()
        
        return predictions_df    
    
    def VAR_predict_train_shuffled(x_train, fitted_model , steps = 1):

        '''
        Use fitted VAR model to make predictions
        on test data in a shuffled manner (use random days,
        do not care about order)

        Params
        ------
        x_train: np.array
        x_test : np.array
        fitted_model: fitted VAR model
        steps: forecast horizon

        Returns
        ------
        predictions_df 

        '''

        predictions = []

        random_idx = list(range(len(x_train) - 1)) 
        random.Random(10).shuffle(random_idx)

        for i in random_idx:
            forecast_input = x_train[i].reshape(steps, len(x_train[i]))
            yhat = fitted_model.forecast(y=forecast_input, steps=steps)
            predictions.append(yhat)

        predictions_array = [item for sublist in predictions for item in sublist]
        predictions_df = pd.DataFrame(predictions_array)

        idx = [x+1 for x in random_idx]
        predictions_df.index = idx 
        predictions_df = predictions_df.sort_index()

        return predictions_df   


    def VAR_predict_train_rolling(self, x_train, model, lag=1, steps=1):
    
        ''' 
        fit VAR model on train set and predict 
        on train set using rolling cross validation

        Params
        ---------
        x_train    : np.array with train data
        lag        : lag to use in VAR model
        steps      : forecast horizon

        Returns
        ---------
        predictions_array: array with predictions
        
        '''
    
        lag = lag

        history_train =  [x for x in x_train[:lag]]
        predictions = list()

        for i in range(lag, len(x_train)): 

            forecast_input = np.array(history_train[-lag:]).reshape(1, x_train.shape[1])
            yhat = model.forecast(y=forecast_input, steps=steps) #forecast
            predictions.append(yhat)

            obs = x_train[i]
            history_train.append(obs)

        predictions_array = [item for sublist in predictions for item in sublist]

        return predictions_array
    
    
    def VAR_predict_test_rolling(self, x_train, x_test, lag=1, steps=1):
    
        ''' 
        fit VAR model on train set and predict 
        on test set using rolling cross validation

        Params
        ---------
        x_train    : np.array with train data
        x_test     : np.array with test data
        lag        : lag to use in VAR model
        steps      : forecast horizon

        Returns
        ---------
        fitted_var_model : fitted model object
        predictions_array: array with predictions 
        
        '''
        
        lag = lag

        model = VAR(x_train)
        fitted_var_model = model.fit(lag)
    
        history = [x for x in x_train]
        predictions = list()
        forecast_input = x_train[-lag:]
        yhat = fitted_var_model.forecast(y=forecast_input, steps=steps)

        predictions.append(yhat)
        history.append(x_test[0]) 

        for i in range(1, len(x_test)):

            model = VAR(history)
            fitted_var_model = model.fit(lag)

            yhat = fitted_var_model.forecast(y=np.array(history[-lag:]), steps=steps)
            predictions.append(yhat)

            obs = x_test[i]
            history.append(obs)

        predictions_array = [item for sublist in predictions for item in sublist]

        return fitted_var_model, predictions_array
