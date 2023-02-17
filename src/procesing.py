import os
os.chdir('/Users/zaretcardenas/Documents/MaestriaDS/arquitectura/demo/zaret_cardenas_tarea04/')

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
from src.utils import saving_results

### importar el logger desde cada modulo 
import logging
logger = logging.getLogger(__name__)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

#drop_col_feat drop_col_prepro


##-----------------------------EDA  

def loading_data():
    

    train_data = pd.read_csv("house-prices-data/train.csv")
    test_data = pd.read_csv("house-prices-data/test.csv")
    test_ids = test_data['Id']
    logger.info('Shape after loading data ={}'.format(train_data.shape[0]))
   
    
    try:
        
        train_data[drop_col_feat]
        train_data[drop_col_prepro]
        train_data[ordinal_col]
        train_data[level_col]
        
        test_data[drop_col_feat]
        test_data[drop_col_prepro]
        test_data[ordinal_col]
        test_data[level_col]
        return train_data,test_data,test_ids
 
    except:
        logger.info('Variables required in utils not found in train or test ')
        

def eda(train_data,test_data):

    

    
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
    
    logger.info('Saving the graphs in images/ heat_map.png and boxplot.png')

    


##-----------------------------proprocesing 

def preprocesing(train_data,test_data,variable_fillna_esp,ordinal_col,level_col):
    
 
    ### Filling with 'No' a special set of variables( variable_fillna_esp)
    train_data[variable_fillna_esp]=train_data[variable_fillna_esp].fillna("No")
    test_data[variable_fillna_esp]=test_data[variable_fillna_esp].fillna("No")
    
    
    
     
    ##Calling function to drop columns and filling na values 
    fill_all_missing_values(train_data)
    fill_all_missing_values(test_data)
    
    ##drop columns in preprocesing 
    logging.warning('This variables were been erased {}'.format(drop_col_prepro))
   
    train_data.drop(drop_col_prepro, axis=1, inplace=True)
    test_data.drop(drop_col_prepro, axis=1, inplace=True)
    
   
 
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
    
    logger.info('Shape after preprocessing ={}'.format(train_data.shape[0]))
   
      
    return train_data, test_data


########---------------------featurizing 


def featurizing(train_data,test_data,drop_col_feat):

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
    
  
    logging.warning('This variables were been erased {}'.format(drop_col_feat))
   
    
    train_data.drop(drop_col_feat, axis=1, inplace=True)
    test_data.drop(drop_col_feat, axis=1, inplace=True)

    logger.info('Shape after featurizing ={}'.format(train_data.shape[0]))
   
    return train_data, test_data


def trainning(train_data_f,test_data,test_ids,node,saving,target) :
    
    y = train_data_f[target]
    X = train_data_f.drop([target], axis=1)
    
    
    # in this case the nodes were in a loop but it only contains one node, so i decide break 
    #that for ,in case that we want to use other hyperparameter I´ll proprose a pipeline but for this example
    #is not necesary 
    
    # Number of nodes to check 
    model = RandomForestRegressor(max_leaf_nodes=node,)
    model.fit(X, y)
    score = cross_val_score(model, X, y, cv=10)
  
    logger.info('Score mean of cross validation ={}'.format(score.mean()))
  
    
    price = model.predict(test_data)
    
    
    
    if saving==True:
       saving_results(test_ids,price)
    else :
       print('Predictors not saving')
       logger.info('The results were not saved')
     
    return price
