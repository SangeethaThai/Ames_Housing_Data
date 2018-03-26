from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RandomizedLasso
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE
import numpy as np
import pandas as pd

# LOAD all data into single data frame

def load_train_data() :
    train_df = pd.read_csv('data/train.csv')    
    return train_df 


def load_test_data() :
    test_df = pd.read_csv('data/test.csv')    
    return test_df 


def clean_data(housing_df) :
    # remove cols
    housing_df.set_index('Id', inplace=True)
    
    drop_cols = ['PoolArea', 'PoolQC']
    housing_df.drop(drop_cols, axis=1, inplace=True)
    
    # replace pool features with HasPool
    #housing_df["HasPool"]
    
    # impute na with mean for certain features
    housing_df.LotFrontage.fillna(housing_df.LotFrontage.mean(), inplace=True)
    housing_df.MasVnrArea.fillna(housing_df.MasVnrArea.mean(), inplace=True)
    housing_df.GarageYrBlt.fillna(housing_df.GarageYrBlt.mean(), inplace=True)

    # fill blank categorical features with appropriate values
    #housing_df.Alley.cat.add_categories("No alley access")
    housing_df.Alley.fillna("No alley access", inplace=True)
    housing_df.BsmtQual.fillna("No Basement", inplace=True)
    housing_df.BsmtExposure.fillna("No Basement", inplace=True)
    housing_df.BsmtFinType1.fillna("No Basement", inplace=True)
    housing_df.BsmtFinType2.fillna("No Basement", inplace=True)
    housing_df.FireplaceQu.fillna("No Fireplace", inplace=True)
    housing_df.GarageType.fillna("No Garage", inplace=True)
    housing_df.GarageFinish.fillna("No Garage", inplace=True)
    housing_df.GarageQual.fillna("No Garage", inplace=True)
    housing_df.GarageCond.fillna("No Garage", inplace=True)
    housing_df.Fence.fillna("No Fence", inplace=True)
    housing_df.MiscFeature.fillna("None", inplace=True)
    housing_df.Electrical.fillna("No Electrical", inplace=True)
    housing_df.MasVnrType.fillna("None", inplace=True)
    
    # convert object types to category
    for column in housing_df.select_dtypes(['object']).columns:
        housing_df[column] = housing_df[column].astype('category')
    
    housing_df.dropna(inplace=True)
    return housing_df
    

def scale_numeric_features(housing_df) :
    all_features = housing_df.iloc[:,:-1]
    numeric_features = all_features.select_dtypes(include=['float', 'int'])
    num_log_df = np.log(numeric_features + 1)
    num_log_sc_df = (num_log_df - num_log_df.mean())/(2*num_log_df.std())
    all_features.update(num_log_sc_df)
    return all_features
    
    
def one_hot_encode_categorical_features(housing_df) :
    return pd.get_dummies(housing_df)


def eda_selected_features() :
    return ['GrLivArea', '1stFlrSF', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt',   # numeric features
            'Utilities_AllPub', 'Street_Pave', 'Condition2_Norm', 'RoofMatl_CompShg', 'Heating_GasA'   # categorical features
           ]

def lasso_selected_features(features, target, n_features) :
    random_lasso = Lasso()
    random_lasso.fit(features, target)
    random_lasso.coef_

    df = pd.DataFrame()
    df["colnames"] = features.columns
    df["coefs"] = np.abs(random_lasso.coef_)
    df = df.sort_values(by='coefs', ascending=False)
    df = df.head(n_features)   
    return df["colnames"]


#Recursive Feature Elimination (RFE)
def rfe_linear_selected_features(features, target, n_features) :   
    model = LinearRegression()
    rfe = RFE(model, n_features)  # create the RFE model and select n attributes
    rfe = rfe.fit(features, target)
    
    df = pd.DataFrame()
    df["colnames"] = features.columns
    df["support"] = rfe.support_
    df["ranking"] = rfe.ranking_
    selected_df = df.loc[df["support"] == True]
    return selected_df["colnames"]








