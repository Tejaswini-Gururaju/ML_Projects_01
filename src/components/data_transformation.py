import numpy as np
import pandas as pd
import sys
from dataclasses import dataclass
import os
from src.utils import save_object


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path=os.path.join('artifact',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        ''' This Fun is responsible for data trans'''
        try:
            numerical_colm=["writing_score","reading_score"]
            categorical_colm=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_courses",

            ]
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",SimpleImputer())
                ]
            ),
            categorical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler())
                ]
            ),
            logging.info("numerical columns standard scaling completed")
            logging.info("categorical columns encoding completed")

            preprocessor=ColumnTransformer(
                # logging.info("entered coulmn trans")
                transformers=[
                    ("num_pipeline",numerical_pipeline,numerical_colm),
                    ("cat_pipeline",categorical_pipeline,categorical_colm),

                ]
                
            )
            logging.info("exited the column trans")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("read train and test data completed")

            logging.info("obtaining preprocessig objects")

            preprocessing_obj = self.get_data_transformer_obj()

            target_colmns = "math_score"
            numerical_colm=["writing_score","reading_score"]
            logging.info("target column specified")

            input_features_train_df = train_df.drop(columns=[target_colmns],axis=1)
            logging.info("independent train feature ")

            target_feature_train_df=train_df[target_colmns]

            logging.info("got our input feature")

            input_features_test_df = test_df.drop(columns=[target_colmns],axis=1)
            target_feature_test_df=test_df[target_colmns]

            logging.info("applying preprocessing object on training df and test df")
            
            try :
                input_train_trans = preprocessing_obj.fit_transform(input_features_train_df)
                logging.info("input train data transformed")

                return input_train_trans
            
            except Exception as e:
                return CustomException(e,sys)

            
            input_test_trans = preprocessing_obj.transform(input_features_test_df)
            logging.info("input test data transformed")

            train_arr = np.c_[input_train_trans,np.array(target_feature_train_df)]
            test_arr = np.c_[input_test_trans,np.array(target_feature_test_df)]

            logging.info("saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj
            )


            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path,

            )




        except Exception as e:
            return CustomException(e,sys)


   