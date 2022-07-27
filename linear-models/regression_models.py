import pandas as pd
import matplotlib.pyplot as plt
import warnings
import numpy as np
import seaborn as sns
import os
import sys

#sklearn
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNetCV, RidgeCV, LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import TruncatedSVD

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler



class OLS_model:
    
    def __init__(self, n_features):
        
        self.n_features = n_features
        
    def test_model(self, train, test):
        
        predictions_df = pd.DataFrame()
        
        for i in range(0, self.n_features):
            try:

                y_train = train.filter(like = 'lag0')
                y_train = y_train.iloc[:, i] 

                X_train = train.drop(train.filter(regex='lag0').columns, axis=1)
                X_train = sm.add_constant(X_train, has_constant='add')
                
                X_test = test.drop(test.filter(regex='lag0').columns, axis=1)
                X_test = sm.add_constant(X_test, has_constant='add')

                model = sm.OLS(y_train, X_train).fit()
                prediction = model.predict(X_test)

                predictions_df[i] = prediction

            except:
                predictions_df[i] = 'estimation infeasible'
        
        return predictions_df
    
    
class PoissonModel:
    
    def __init__(self, n_features):
        
        self.n_features = n_features
        
    def test_model(self, train, train_scaled, test, test_scaled, regularized = False):
        
        predictions_df = pd.DataFrame()
        
        for i in range(0, self.n_features):

            try:

                y_train = train.filter(like = 'lag0')
                y_train = y_train.iloc[:, i].values

                X_train = train_scaled.drop(train_scaled.filter(regex='lag0').columns, axis=1)
                X_train = sm.add_constant(X_train, has_constant='add')
                
                X_test = test_scaled.drop(test_scaled.filter(regex='lag0').columns, axis=1)
                X_test = sm.add_constant(X_test, has_constant='add')
                
                model = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
                prediction = model.predict(X_test)
                    
                predictions_df[i] = prediction

            except:
                predictions_df[i] = 'estimation infeasible'
        
        return predictions_df
    
    
class NB2Model:
    
    
    '''
    
    Calculate mu parameter (counts frequency) using Poisson regression 
    and use mu in negative binomial regression model 
    
    Params:
    train : count dataframe
    test  : count dataframe
    
    '''
    
    def __init__(self, n_features):
        
        self.n_features = n_features

    def calculate_poisson_mu(self, y, X):

        model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        poisson_mu = model.mu

        lambda_vector = ((y - poisson_mu)**2 - poisson_mu)/poisson_mu
        aux_olsr_results = sm.OLS(lambda_vector, poisson_mu).fit()

        return aux_olsr_results

    def test_model(self, train, test):

        predictions_df = pd.DataFrame()
        
        for i in range(0, self.n_features):
            
            try:
                y_train = train.filter(like = 'lag0')
                y_train = y_train.iloc[:, i] 

                X_train = train.drop(train.filter(regex='lag0').columns, axis=1)
                X_train = sm.add_constant(X_train, has_constant='add')
                
                X_test = test.drop(test.filter(regex='lag0').columns, axis=1)
                X_test = sm.add_constant(X_test, has_constant='add')
                
                aux_olsr_results = self.calculate_poisson_mu(y_train, X_train)
                model = sm.GLM(y_train, X_train, family=sm.families.NegativeBinomial(alpha=aux_olsr_results.params[0])).fit()
                
                prediction = model.predict(X_test)
                predictions_df[i] = prediction

            except:
                predictions_df[i] = 'estimation infeasible'
        
        return predictions_df
    
    
class SVD_OLS_MODEL:
    
    '''
    
    Use SVD to reduce dimensionality of regressors 
    and use SVD components as regerssors for y
    
    Params:
    train : scaled train data
    test  : scaled test data
    
    '''
        
    def __init__(self, n_features, n_components):
        
        self.n_features = n_features
        self.n_components = n_components

    def calculate_svd_components(self, X):
        
        svd = TruncatedSVD(n_components=self.n_components, n_iter=7, random_state=42)
        svd_X = svd.fit_transform(X)
        svd_X = sm.add_constant(svd_X)
        
        return svd_X

    def test_model(self, train, test):

        predictions_df = pd.DataFrame()
        
        for i in range(0, self.n_features):
            
            try:
                y_train = train.filter(like = 'lag0')
                y_train = y_train.iloc[:, i] 

                X_train = train.drop(train.filter(regex='lag0').columns, axis=1)
                X_train_svd = self.calculate_svd_components(X_train)
                X_train_svd = sm.add_constant(X_train_svd, has_constant='add')
                
                X_test = test.drop(test.filter(regex='lag0').columns, axis=1)
                X_test_svd = self.calculate_svd_components(X_test)
                X_test_svd = sm.add_constant(X_test_svd, has_constant='add')
                
                model = sm.OLS(y_train, X_train_svd).fit()
                
                prediction = model.predict(X_test_svd)
                predictions_df[i] = prediction

            except:
                predictions_df[i] = 'estimation infeasible'
        
        return predictions_df
    
    
    
class RidgeModel:
    
    def __init__(self, n_features):
        
        self.n_features = n_features
        
    def test_model(self, train, test):
        
        predictions_df = pd.DataFrame()
        
        for i in range(0, self.n_features):

            try:

                y_train = train.filter(like = 'lag0')
                y_train = y_train.iloc[:, i] 

                X_train = train.drop(train.filter(regex='lag0').columns, axis=1)
                
                X_test = test.drop(test.filter(regex='lag0').columns, axis=1)
                
                alphas = [1e-1, 1e-2, 1e-3, 1e-4]

                cv = RepeatedKFold(n_repeats=3, n_splits=5, random_state=1)
                model = RidgeCV(alphas = alphas,
                                normalize=False,
                                cv = cv).fit(X_train, y_train)
                
                prediction = model.predict(X_test)
                predictions_df[i] = prediction.tolist()

            except:
                predictions_df[i] = 'estimation infeasible'
        
        return predictions_df
    
    
class LassoModel:
    
    def __init__(self, n_features):
        
        self.n_features = n_features
        
    def test_model(self, train, test):
        
        predictions_df = pd.DataFrame()
        
        for i in range(0, self.n_features):

            try:

                y_train = train.filter(like = 'lag0')
                y_train = y_train.iloc[:, i] 

                X_train = train.drop(train.filter(regex='lag0').columns, axis=1)
                X_test = test.drop(test.filter(regex='lag0').columns, axis=1)
                
                alphas = [1e-1, 1e-2, 1e-3, 1e-4]

                cv = RepeatedKFold(n_repeats=3, n_splits=5, random_state=1)
                model = LassoCV(alphas = alphas,
                                normalize=False,
                                cv = cv).fit(X_train, y_train)
                
                prediction = model.predict(X_test)
                predictions_df[i] = prediction.tolist()

            except:
                predictions_df[i] = 'estimation infeasible'
        
        return predictions_df
    
class ENETModel:
    
    def __init__(self, n_features):
        
        self.n_features = n_features
        
    def test_model(self, train, test):
        
        predictions_df = pd.DataFrame()
        
        for i in range(0, self.n_features):

            try:

                y_train = train.filter(like = 'lag0')
                y_train = y_train.iloc[:, i] 

                X_train = train.drop(train.filter(regex='lag0').columns, axis=1)
                X_test = test.drop(test.filter(regex='lag0').columns, axis=1)
                
                l1_ratio = [.1, .5, .7, .9, .95]
                alphas = [1e-2, 1e-3, 1e-4]

                cv = RepeatedKFold(n_repeats=3, n_splits=5, random_state=1)
                model = ElasticNetCV(l1_ratio = l1_ratio,
                                     alphas = alphas,
                                     normalize=False,
                                     cv = cv,
                                     random_state=1).fit(X_train, y_train)
                
                prediction = model.predict(X_test)
                predictions_df[i] = prediction.tolist()

            except:
                predictions_df[i] = 'estimation infeasible'
        
        return predictions_df