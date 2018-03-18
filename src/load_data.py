import numpy as np
import pandas as pd

# LOAD all data into single data frame
zoning_df = pd.read_csv('data/zoning.csv')
listing_df = pd.read_csv('data/listing.csv')
sale_df = pd.read_csv('data/sale.csv')

housing_df = pd.merge(zoning_df, listing_df, left_on="Id", right_on="Id")
housing_df = pd.merge(housing_df, sale_df, left_on="Id", right_on="Id")

housing_df.set_index("Id", inplace=True)

# filter - choose only the top 10 features selected for PCA Analysis

numerical_features = housing_df.loc[:, ['GrLivArea', 'FirstFlrSF', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt']]
categorical_features = housing_df.loc[:, ['Utilities', 'Street', 'Condition2', 'RoofMatl', 'Heating']]


for column in categorical_features.select_dtypes(['object']).columns:
    categorical_features[column] = categorical_features[column].astype('category')


housing_df.GarageYrBlt.fillna(housing_df.GarageYrBlt.mean(), inplace=True)


numerical_features.dropna(inplace=True)

numeric_sc_df = (numerical_features - numerical_features.mean())/numerical_features.std()
numeric_log_df = np.log(numerical_features + 1)
numeric_log_sc_df = (numerical_features - numerical_features.mean())/numerical_features.std()

outliers = [198, 524, 1183, 1299, 186, 692, 770, 
            179, 225, 804, 889, 1387, 497]
numeric_final_df = numeric_log_sc_df.drop(outliers, axis=0)
categorical_ouliers_removed_df = categorical_features.drop(outliers, axis=0)