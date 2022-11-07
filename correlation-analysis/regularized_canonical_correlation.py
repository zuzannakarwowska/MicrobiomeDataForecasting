from collections import Counter
import pandas as pd
import numpy as np
import os
import sys

from skbio.stats.composition import clr
from scipy import stats
import rcca
from sklearn.preprocessing import StandardScaler

MODULE_PATH = os.path.abspath('/storage/zkarwowska/predicting-microbiome-in-time/data-processing/data-preparation/') 
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)
    
from processing import MicrobiomeDataPreprocessing, CLRTransformer, MicrobiomeTraintestSplit
from interpolation import Interpolation


MALE_FILE = '/storage/zkarwowska/microbiome-interactions/datasets/processed/ready_datasets_no_rarefaction/male_assigned_sample_names.csv'

pseudo=0.5
DF = pd.read_csv(MALE_FILE, index_col = [0]).dropna()
DF.index  = DF.index.astype(int)

#interpolate using pchip
interpolator = Interpolation()
INTERPOLATED_DF = interpolator.interpolate_pchip(DF) 

n = 200
idx = pd.DataFrame(INTERPOLATED_DF.mean()).sort_values(by = [0]).iloc[-n:].index
other_cols = pd.DataFrame(INTERPOLATED_DF.mean()).sort_values(by = [0]).iloc[:-n].index

other_col = INTERPOLATED_DF[other_cols].sum(axis=1)
INTERPOLATED_DF_OTHER = INTERPOLATED_DF[idx] 
INTERPOLATED_DF_OTHER['other'] = other_col.values
INTERPOLATED_DF_OTHER = INTERPOLATED_DF_OTHER + pseudo

clr_df = pd.DataFrame(clr(INTERPOLATED_DF_OTHER), columns = INTERPOLATED_DF_OTHER.columns)

#prepare lagged dataset
processing = MicrobiomeDataPreprocessing()
maxlag = 9

df_supervised = processing.make_supervised(clr_df, maxlag=maxlag)

def run_cca(df_supervised, lag):
    
    X1 = df_supervised.filter(like = 'lag0').values  
    X2 = df_supervised.filter(like = 'lag{}'.format(lag))

    scaler=StandardScaler()
    X1_sc = scaler.fit_transform(X1)
    X2_sc = scaler.fit_transform(X2)

    # search for optimal parameters for CCA: regularization strength and number of components
    def cca_gridsearch(X1, X2):

        BestReg = []
        BestCpp = []

        iterations=1

        for i in range(iterations):

            regs = [1e-3, 1e-2, 1e-1, 1]
            numCCs = [3, 5, 10, 20, 50, 100]

            ccaCV = rcca.CCACrossValidate(numCCs=numCCs, regs=regs)

            ccaCV.train([X1_sc, X2_sc])

            best_comp = ccaCV.best_numCC
            best_reg = ccaCV.best_reg

            BestReg.append(best_reg)
            BestCpp.append(best_comp)

        best_reg_param = pd.DataFrame.from_dict(Counter(BestReg), orient='index').reset_index().sort_values(by = [0]).tail(1)['index']
        best_cpp_param = pd.DataFrame.from_dict(Counter(BestCpp), orient='index').reset_index().sort_values(by = [0]).tail(1)['index']


        return best_reg_param, best_cpp_param

    best_reg_param, best_cpp_param  = cca_gridsearch(X1, X2)
    
    #run cca with best components
    full_cca = rcca.CCA(numCC=int(best_cpp_param), reg=best_reg_param.tolist(), verbose=False)
    full_cca.train([X1_sc, X2_sc])

    #extract and save canonical correlations for each canonical variates pair
    canonical_corr_df = pd.DataFrame(full_cca.cancorrs, columns = ['canonical_correlation'])
    
    path = '/storage/zkarwowska/predicting-microbiome-in-time/correlation-analysis/canonical_correlation/results/'
    canonical_corr_df.to_csv(path + 'canonical_corr_df_lag{}.csv'.format(lag), index=False)
    
    #extract and save each canonical variate

    x1_comp = full_cca.comps[0]
    x2_comp = full_cca.comps[1]
    x1_comp_df = pd.DataFrame(x1_comp, columns = ['CC{}'.format(i) for i in range(x1_comp.shape[1])])
    x2_comp_df = pd.DataFrame(x2_comp, columns = ['CC{}'.format(i) for i in range(x1_comp.shape[1])])

    x1_comp_df.to_csv(path + 'x1_comp_lag{}.csv'.format(lag), index=False)
    x2_comp_df.to_csv(path + 'x2_comp_lag{}.csv'.format(lag), index=False)
    
    #calculate and save loadings of features from X2 
    features_correlation = pd.DataFrame([stats.pearsonr(full_cca.comps[0][:, 0], X2_sc[:, i])[0] for i in range(X2_sc.shape[1])], columns = ['loadings'])
    features_correlation['feature'] = X2.columns
    features_correlation.to_csv(path + 'features_loadings_lag{}.csv'.format(lag), index=False)

    
for i in range(1, maxlag+1):
    run_cca(df_supervised, i)