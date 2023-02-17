import os
os.chdir('/Users/zaretcardenas/Documents/MaestriaDS/arquitectura/demo/zaret_cardenas_tarea04/')

import os
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
## variables saved in utils
from src.utils import variable_fillna_esp
from src.utils import drop_col_prepro
from src.utils import ordinal_col
from src.utils import level_col
from src.utils import drop_col_feat
from src.utils import fill_all_missing_values
from src.utils import saving_results
## Steps saved as function in the .py procesing 
from src.procesing import eda
from src.procesing import preprocesing
from src.procesing import featurizing
from src.procesing import trainning
from src.procesing import loading_data
#import
import logging
      

#cwd = os.getcwd()
#os.chdir(cwd+'/')
  

# eda(train_data,test_data)



def main():
    logging.basicConfig(filename='./logs/results.log', level=logging.INFO,  filemode='w')
    
    
    try :
        logging.info('Started loading data')
        
        train_data,test_data,test_ids=loading_data()
        
        eda(train_data,test_data)
        
        logging.info('Started preprocess')
      
        train_data_p,test_data_p=preprocesing(train_data,test_data,variable_fillna_esp,ordinal_col,level_col)

        
        logging.info('Started Featurizing')
      
        train_data_f,test_data_f=featurizing(train_data_p,test_data_p,drop_col_feat)

        
        logging.info('Started Trainning')
        
        price=trainning(train_data_f,test_data,test_ids,node=250,saving=True,target='SalePrice')

    except:
        logging.info('Variables required in utils not found in train or test ')

    
   
    
    

if __name__ == '__main__':
    main()

