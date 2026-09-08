import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_objects(self): # we are basicaly preprocessing the data before sending it to the model
        try:
            # here we are hard coding our datas header as we mostly know our data beforehand because of EDA
            numerical_columns=['writing_score','reading_score']
            categorical_columns=[
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            # now we are making the datacleaning pipeline which will clean the data as it comes
            num_pipeline=Pipeline(
                steps=[
                    # we use imputer to replace missing value with NaN or NULL
                    ('imputer',SimpleImputer(strategy='median')), # here we are replacing it with the median
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info('Numerical coulumns Stabdard scaling completed')

            # we added an extra step in cat_features that is one hot encoder
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info('Categorical coulumns encoding completed')

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                    # here we are changed our transformer to pipeline
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Read the train and test data completed')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj=self.get_data_transformer_objects() # we are creating the whole preprocessing function into an object

            target_column_name='math_score'

            # we are spliting data again to dependent and independent features for both the splits
            input_feature_train_df=train_df.drop(columns=[target_column_name])
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name])
            target_feature_test_df=test_df[target_column_name]

            logging.info('Applying preprocessing objects on training and testing df')

            # now we are sending the train and test data to get preprocessed
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info('Saved preprocessing object')

            # here we are saving our model which we created as pickle file using util.py file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
