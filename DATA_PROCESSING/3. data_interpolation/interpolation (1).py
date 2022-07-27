from scipy import interpolate, stats
import pandas as pd
from scipy.interpolate import pchip_interpolate, BSpline, splev

class Interpolation:
    
    '''
    Identify which datapoints are missing and interpolate each columns
    separatery using linear, bspline, nearest of PCHip interpolations.
    
    
    Params
    ---------
    df - dataframe that you wish to interpolate. What is important is that your index
    must show that your dataframe misses timepoints. 
    
    Ex. your dataframe index is 0, 1, 3, 5, and 8 so you will interpolate
    timepoints 2, 4, 6 and 7. 
    
    
    Output
    ---------
    
    interpolated_df - interpolated dataframe
    
    
    
    How to use this class:
    ---------
    interpolator = Interpolation()
    linear_interpolated_df = interpolator.interpolate_linear(df)
    pchip_interpolated_df = interpolator.interpolate_pchip(df)
    bspline_interpolated_df = interpolator.interpolate_nearest(df, 2)
    
    '''
    
    def apply_interpolation(self, df, interpolation_function):

        INTERPOLATED_COLUMNS = []
        for col in df.columns:
            interpolated_col = interpolation_function(col, df)
            INTERPOLATED_COLUMNS.append(interpolated_col)
        INTERPOLATED_DF = pd.concat(INTERPOLATED_COLUMNS, axis=1)

        return INTERPOLATED_DF
    
    def apply_bspline_interpolation(self, df, interpolation_function, degree=1):
    
        INTERPOLATED_COLUMNS = []
        for col in df.columns:
            interpolated_col = interpolation_function(col, df, degree)
            INTERPOLATED_COLUMNS.append(interpolated_col)
        INTERPOLATED_DF = pd.concat(INTERPOLATED_COLUMNS, axis=1)

        return INTERPOLATED_DF

    
    def linear_interpolation(self, col, masked_df):
    
        df_interpolated = pd.DataFrame(index = masked_df.index)

        tmp = masked_df[col]
        base_nodes =  tmp.dropna().index 
        interpolated_nodes = tmp[tmp.isna()].index 

        f = interpolate.interp1d(base_nodes,
                                 tmp.dropna().values,
                                 kind='linear')

        new_y = f(interpolated_nodes.astype(int))

        name = str(col)
        df_interpolated.loc[base_nodes, name] = tmp.dropna().values
        df_interpolated.loc[interpolated_nodes, name] = new_y

        return df_interpolated
    
    
    def pchip_interpolation(self, col, masked_df):
    
        df_interpolated = pd.DataFrame(index = masked_df.index)

        tmp = masked_df[col]
        base_nodes =  tmp.dropna().index 
        interpolated_nodes = tmp[tmp.isna()].index 

        y = pchip_interpolate(base_nodes,
                              tmp.dropna().values,
                              interpolated_nodes)


        name = str(col)
        df_interpolated.loc[base_nodes, name] = tmp.dropna().values
        df_interpolated.loc[interpolated_nodes, name] = y
    
        return df_interpolated
    
    def bspline_interpolation(self, col, masked_df, degree=1):
    
        df_interpolated = pd.DataFrame(index = masked_df.index)

        tmp = masked_df[col]
        base_nodes =  tmp.dropna().index 
        interpolated_nodes = tmp[tmp.isna()].index 

        f = BSpline(base_nodes,tmp.dropna().values, degree)
        new_y = f(interpolated_nodes.astype(int))

        name = str(col)
        df_interpolated.loc[base_nodes, name] = tmp.dropna().values
        df_interpolated.loc[interpolated_nodes, name] = new_y

        return df_interpolated
    
    def nearest_interpolation(self, col, masked_df):
    
        df_interpolated = pd.DataFrame(index = masked_df.index)

        tmp = masked_df[col]
        base_nodes =  tmp.dropna().index #wezlowe
        interpolated_nodes = tmp[tmp.isna()].index #to uzupelniamy

        f = interpolate.interp1d(base_nodes,
                                 tmp.dropna().values,
                                 kind='nearest')
        new_y = f(interpolated_nodes.astype(int))

        name = str(col)
        df_interpolated.loc[base_nodes, name] = tmp.dropna().values
        df_interpolated.loc[interpolated_nodes, name] = new_y

        return df_interpolated
    
    def prepare_data_for_interpolation(self, df):
        
        start_df = df.iloc[0].name
        end_df = df.iloc[-1].name

        full = list(range(start_df, end_df)) 
        missing_tpoints = list(set(full) - set(df.index.astype(int)))
        missing_df = df.reindex(df.index.union(missing_tpoints))

        return missing_df

    def interpolate_linear(self, df):
        
        df = self.prepare_data_for_interpolation(df)
        df_interpolated = self.apply_interpolation(df, self.linear_interpolation)
        
        return df_interpolated.astype(float)
        
    def interpolate_pchip(self, df):
        
        df = self.prepare_data_for_interpolation(df)
        df_interpolated = self.apply_interpolation(df, self.pchip_interpolation)
        
        return df_interpolated.astype(float)
    
    def interpolate_nearest(self, df):
        
        df = self.prepare_data_for_interpolation(df)
        df_interpolated = self.apply_interpolation(df, self.nearest_interpolation)
        
        return df_interpolated.astype(float)
    
    
    def interpolate_bspline(self, df, degree):
        
        df = self.prepare_data_for_interpolation(df)
        df_interpolated = self.apply_bspline_interpolation(df, self.bspline_interpolation, degree)
        
        return df_interpolated.astype(float)