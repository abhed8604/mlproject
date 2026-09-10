import os
import sys
import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test,models,params): # we are creating an function which will evaluate all the models in models dict for our data
    try:
        report={} # we made our report dict to store our result

        for i in range(len(list(models))):
            model=list(models.values())[i] # here we select one model
            para=params[list(models.keys())[i]]

            gs = GridSearchCV( # doing grid search for hyperparameter tuning
                estimator=model,
                param_grid=para,
                cv=3,
            )
            gs.fit(X_train,y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train) # we are fitting the best model we found from grid search CV

            # we get our predicted values for our test array and then get r2 score for it 
            y_test_pred=model.predict(X_test)
            test_model_score=r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]]=test_model_score # we save our result 

        return report
        
    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path): # we will use this function to load the model from pickle file
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
