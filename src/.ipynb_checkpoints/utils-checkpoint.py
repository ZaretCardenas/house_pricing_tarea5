import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error             

## In this file is allocated the values constants that wer use in trainning 

variable_fillna_esp=['FireplaceQu','BsmtQual','BsmtCond','BsmtFinType1','BsmtFinType2']

    

drop_col_prepro = ['Id', 'Alley', 'PoolQC', 'MiscFeature', 'Fence', 'MoSold', 'YrSold', 'MSSubClass',
            'GarageType', 'GarageArea', 'GarageYrBlt', 'GarageFinish', 'YearRemodAdd', 'LandSlope',
            'BsmtUnfSF', 'BsmtExposure', '2ndFlrSF', 'LowQualFinSF', 'Condition1', 'Condition2', 'Heating',
             'Exterior1st', 'Exterior2nd', 'HouseStyle', 'LotShape', 'LandContour', 'LotConfig', 'Functional',
             'BsmtFinSF1', 'BsmtFinSF2', 'FireplaceQu', 'WoodDeckSF', 'GarageQual', 'GarageCond', 'OverallCond'
           ] 

ordinal_col=['BsmtQual','BsmtCond','ExterQual','ExterCond','KitchenQual','Utilities','MSZoning','PavedDrive',
           'Electrical','BsmtFinType1','BsmtFinType2',
           'Foundation', 'Neighborhood','MasVnrType','SaleCondition','RoofStyle','RoofMatl']
level_col = ['Street' ,'BldgType','CentralAir' ] # 'SaleType'

drop_col_feat = ['OverallQual', 
            'ExterCond', 'ExterQual',
            'BsmtCond', 'BsmtQual',
            'BsmtFinType1', 'BsmtFinType2',
            'HeatingQC',
            'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
            'BsmtFullBath', 'BsmtHalfBath',
            'FullBath', 'HalfBath','SaleType'
           ]

def fill_all_missing_values(data):
    for col in data.columns:
        if((data[col].dtype == 'float64') or (data[col].dtype == 'int64') ):
            data[col].fillna(data[col].mean(),inplace=True)
            
        else:
             data[col].fillna(data[col].mode()[0], inplace=True)
def saving_results(test_ids,price):
    print('Saving result/submission.csv ')
    submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": price,

    })

    submission.to_csv("result/submission.csv", index=False)
    submission.sample(10)

        