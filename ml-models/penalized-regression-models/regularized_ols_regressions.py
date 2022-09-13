import pandas as pd
import numpy as np

#sklearn
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNetCV, RidgeCV, LassoCV
from sklearn.model_selection import cross_val_score


class RidgeModel:
    
    
    '''
    Use ridge regression with built-in cross-validation 
    to find optimal lambda params
    '''
    
    
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