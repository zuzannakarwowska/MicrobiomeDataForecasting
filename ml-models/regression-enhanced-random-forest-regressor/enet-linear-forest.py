#base
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

import os
import sys

import pickle


from scipy.interpolate import pchip_interpolate, BSpline, splev
from scipy import interpolate, stats

import skbio
from skbio.stats.composition import clr, alr, alr_inv, ilr, ilr_inv
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from lineartree import LinearForestRegressor

MODULE_PATH = os.path.abspath('/storage/zkarwowska/predicting-microbiome-in-time/data-processing/data-preparation/') 
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)
    
from processing import MicrobiomeDataPreprocessing, CLRTransformer, MicrobiomeTraintestSplit
from interpolation import Interpolation

INPUT_FILE = '/storage/zkarwowska/microbiome-interactions/datasets/processed/ready_datasets_no_rarefaction/male_assigned_sample_names.csv'

DF = pd.read_csv(INPUT_FILE, index_col = [0])
DF.index  = DF.index.astype(int)

#interpolate using pchip
interpolator = Interpolation()
INTERPOLATED_DF = interpolator.interpolate_pchip(DF)

#change to PRC
INTERPOLATED_PRC = INTERPOLATED_DF.div(INTERPOLATED_DF.sum(axis=1), axis=0) + 1e-9

#change to ratio
CLR_DF =  pd.DataFrame(clr(INTERPOLATED_PRC), columns = INTERPOLATED_PRC.columns)

#filter rare bacteria 
processing = MicrobiomeDataPreprocessing()
KEEP_COLUMNS = processing.filter_rare_features(INTERPOLATED_DF, treshold_perc=0.7).columns


FILTERED_CLR_DF = CLR_DF[KEEP_COLUMNS]
FILTERED_COUNT_DF = INTERPOLATED_PRC[KEEP_COLUMNS]

def enet_linear_forest(X, y):
    
    param_grid={
        'base_estimator__alpha':[0.1, 0.5, 1, 2],
        'base_estimator__l1_ratio': [0.2, 0.5, 0.8],
        'n_estimators': [50, 100, 500, 700],
        'max_depth': [10, 20, 30, 50],
        'min_samples_split' : [2, 4, 8, 16, 32],
        'max_features' : ['sqrt']
    }
    
    
    cv = RepeatedKFold(n_repeats=3,
                       n_splits=3,
                       random_state=1)
    
    model = GridSearchCV(
        LinearForestRegressor(ElasticNet(random_state = 0), random_state=42),
        param_grid=param_grid,
        n_jobs=-1,
        cv=cv,
        scoring='neg_root_mean_squared_error'
        )
    
    model.fit(X, y)
    
    return model


lag = 3

PREDICTION_TEST = pd.DataFrame()
PREDICTION_TRAIN = pd.DataFrame()

F_IMPORTANCE = []
COEFF = []

for i in range(len(FILTERED_CLR_DF.columns)):

    denom_idx = i
    denom_name = FILTERED_COUNT_DF.iloc[:, denom_idx].name
    COLS = FILTERED_COUNT_DF.drop(columns=denom_name, axis=1).columns

    FILTERED_ALR_DF = pd.DataFrame(alr(FILTERED_COUNT_DF, denom_idx), columns = COLS)

    processing = MicrobiomeDataPreprocessing()
    CLR_SUPERVISED_DF = processing.make_supervised(FILTERED_CLR_DF, lag).filter(like='lag0')
    ALR_SUPERVISED_DF = processing.make_supervised(FILTERED_ALR_DF, lag)
    ALR_SUPERVISED_DF = ALR_SUPERVISED_DF.drop(columns = ALR_SUPERVISED_DF.filter(like='lag0').columns)

    X_train = ALR_SUPERVISED_DF[:-40]
    X_test = ALR_SUPERVISED_DF[-40:]

    y_train = CLR_SUPERVISED_DF[:-40].iloc[:, denom_idx]
    y_test = CLR_SUPERVISED_DF[-40:].iloc[:, denom_idx]

    model = enet_linear_forest(X_train, y_train)


    prediction_train = model.predict(X_train)
    prediction_test  = model.predict(X_test)
    
    denom_name = FILTERED_COUNT_DF.iloc[:, i].name
    
    filename = 'models/enet-linear-forest_{}.sav'.format(denom_name)
    pickle.dump(model, open(filename, 'wb'))

    PREDICTION_TRAIN[denom_name] = prediction_train 
    PREDICTION_TEST[denom_name] = prediction_test

    rf_importance = pd.DataFrame(model.best_estimator_.feature_importances_, columns = ['feature_importance'])
    rf_importance['features'] = X_train.columns
    rf_importance['target'] = denom_name

    reg_importance = pd.DataFrame(model.best_estimator_.coef_, columns = ['coefficient'])
    reg_importance['features'] = X_train.columns
    reg_importance['target'] = denom_name

    F_IMPORTANCE.append(rf_importance)
    COEFF.append(reg_importance)

F_IMPORTANCE_DF = pd.concat(F_IMPORTANCE)
COEFF_DF = pd.concat(COEFF)


CLR_SUPERVISED_DF[:-40].to_csv('enet_true_train_lag{}.csv'.format(lag))
CLR_SUPERVISED_DF[-40:].to_csv('enet_true_test_lag{}.csv'.format(lag))

PREDICTION_TRAIN.to_csv('enet_pred_train_lag{}.csv'.format(lag))
PREDICTION_TEST.to_csv('enet_pred_test_lag{}.csv'.format(lag))

COEFF_DF.to_csv('enet_coeff_lag{}.csv'.format(lag))
F_IMPORTANCE_DF.to_csv('enet_feature_importance_lag{}.csv'.format(lag))
