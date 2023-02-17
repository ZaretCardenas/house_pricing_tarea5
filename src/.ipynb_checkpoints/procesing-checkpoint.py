import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np

## saving the variables in utils to identify easy changes 
from src.utils import variable_fillna_esp
from src.utils import drop_col_prepro
from src.utils import ordinal_col
from src.utils import level_col
from src.utils import drop_col_feat
from src.utils import fill_all_missing_values








##-----------------------------EDA  

def eda(train_data,test_data):
    
    print("Shape:", train_data.shape)
    print("Duplicated data :", train_data.duplicated().sum())
    print('Saving the graphs heat_map.png and boxplot.png')
    fig, ax = plt.subplots(figsize=(25,10))
    sns.heatmap(data=train_data.isnull(), yticklabels=False, ax=ax)
    
    fig.savefig('images/heat_map.png')

    
    fig, ax = plt.subplots(figsize=(25,10))
    sns.countplot(x=train_data['SaleCondition'])
    sns.histplot(x=train_data['SaleType'], kde=True, ax=ax)
    sns.violinplot(x=train_data['HouseStyle'], y=train_data['SalePrice'],ax=ax)
    sns.scatterplot(x=train_data["Foundation"], y=train_data["SalePrice"], palette='deep', ax=ax)
    plt.grid()
    fig.savefig('images/boxplot.png')

    


##-----------------------------proprocesing 

def preprocesing(train_data,test_data,variable_fillna_esp,ordinal_col,level_col):
    
    print('Preprocesing...')
    ### Filling with 'No' a special set of variables( variable_fillna_esp)
    train_data[variable_fillna_esp]=train_data[variable_fillna_esp].fillna("No")
    test_data[variable_fillna_esp]=test_data[variable_fillna_esp].fillna("No")
    
    
    
     
    ##Calling function to drop columns and filling na values 
    fill_all_missing_values(train_data)
    fill_all_missing_values(test_data)
    
    ##drop columns in preprocesing 
    train_data.drop(drop_col_prepro, axis=1, inplace=True)
    test_data.drop(drop_col_prepro, axis=1, inplace=True)
    
    print(train_data.shape[0])
    print(test_data.shape[0])
    
 
    #---------------ordinal preprocessing 
    
    
    for col in ordinal_col:
        list_cat_test=test_data[col].unique()
        list_cat_train=train_data[col].unique()
        resultList= list(set(list_cat_test) | set(list_cat_train))

    
      
        OE = OrdinalEncoder(categories=[resultList],handle_unknown='use_encoded_value', unknown_value=np.nan)
        train_data[col] = OE.fit_transform(train_data[[col]])
        test_data[col] = OE.transform(test_data[[col]])
    
    #------------- label encoder  preprocessing 
    
    
    encoder = LabelEncoder()
    def encode_catagorical_columns(train, test):
        for col in level_col:
         
            train[col] = encoder.fit_transform(train[col])
            test[col]  = encoder.transform(test[col])
    encode_catagorical_columns(train_data, test_data)
    
    return train_data, test_data


########---------------------featurizing 


def featurizing(train_data,test_data,drop_col_feat):
    print('Feaurizing...')
    train_data['BsmtRating'] = train_data['BsmtCond'] * train_data['BsmtQual']
    train_data['ExterRating'] = train_data['ExterCond'] * train_data['ExterQual']
    train_data['BsmtFinTypeRating'] = train_data['BsmtFinType1'] * train_data['BsmtFinType2']
    
    train_data['BsmtBath'] = train_data['BsmtFullBath'] + train_data['BsmtHalfBath']
    train_data['Bath'] = train_data['FullBath'] + train_data['HalfBath']
    train_data['PorchArea'] = train_data['OpenPorchSF'] + train_data['EnclosedPorch'] + train_data['3SsnPorch'] + train_data['ScreenPorch']
    
    test_data['BsmtRating'] = test_data['BsmtCond'] * test_data['BsmtQual']
    test_data['ExterRating'] = test_data['ExterCond'] * test_data['ExterQual']
    test_data['BsmtFinTypeRating'] = test_data['BsmtFinType1'] * test_data['BsmtFinType2']
    
    test_data['BsmtBath'] = test_data['BsmtFullBath'] + test_data['BsmtHalfBath']
    test_data['Bath'] = test_data['FullBath'] + test_data['HalfBath']
    test_data['PorchArea'] = test_data['OpenPorchSF'] + test_data['EnclosedPorch'] + test_data['3SsnPorch'] + test_data['ScreenPorch']
    
    
    train_data.drop(drop_col_feat, axis=1, inplace=True)
    test_data.drop(drop_col_feat, axis=1, inplace=True)
    print(train_data.shape)
    return train_data, test_data



