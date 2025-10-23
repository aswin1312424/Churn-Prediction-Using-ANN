import os
import pandas as pd
from src.logger import logging
from src.utils import save_object
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")
    cleaned_train_data_path=os.path.join("data/cleaned_data","cleaned_train_data.csv")
    cleaned_test_data_path=os.path.join("data/cleaned_data","cleaned_test_data.csv")

class DataTransformation:
    def __init__(self):
        self.transformation_config=DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        """
        This function is responsible for data transformation
        """
        transformer=ColumnTransformer(
            [
                ("OHE_gen",OneHotEncoder(drop="first"),["Gender"]),
                ("OHE_geo",OneHotEncoder(),["Geography"])
            ],remainder="passthrough"
        )

        pipeline=Pipeline(
            [
                ("transformer",transformer),
                ("standardscaler",StandardScaler())
            ]
        )
        logging.info("preprocessor object created")

        return pipeline
    
    def initiate_data_transformation(self,train_path,test_path):
        train_df=pd.read_csv(train_path)
        test_df=pd.read_csv(test_path)
        logging.info("read train and test dataset")

        target_column="Exited"

        input_feature_train_df=train_df.drop(["RowNumber","CustomerId","Surname",target_column],axis=1)
        target_feature_train_df=train_df[target_column]

        input_feature_test_df=test_df.drop(["RowNumber","CustomerId","Surname",target_column],axis=1)
        target_feature_test_df=test_df[target_column]
        logging.info("dropped unnessacary columns from train and test datasets")

        logging.info("Applying preprocessor object on train and test datasets")
        preprocessor_obj=self.get_data_transformer_obj()

        input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
        input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

        columns=preprocessor_obj.named_steps["transformer"].get_feature_names_out()
        input_train_df=pd.DataFrame(input_feature_train_arr,columns=columns)
        input_test_df=pd.DataFrame(input_feature_test_arr,columns=columns)

        cleaned_train_data=pd.concat([input_train_df,train_df[[target_column]]],axis=1)
        cleaned_test_data=pd.concat([input_test_df,test_df[[target_column]]],axis=1)

        os.makedirs(os.path.dirname(self.transformation_config.cleaned_train_data_path),exist_ok=True)

        cleaned_train_data.to_csv(self.transformation_config.cleaned_train_data_path,index=False,header=True)
        cleaned_test_data.to_csv(self.transformation_config.cleaned_test_data_path,index=False,header=True)

        logging.info("scaled train and test datasets obtained")
        save_object(
            self.transformation_config.preprocessor_obj_file_path,
            preprocessor_obj
        )
        logging.info("saved preprocessor object")
        return(
            self.transformation_config.cleaned_train_data_path,
            self.transformation_config.cleaned_test_data_path,
            self.transformation_config.preprocessor_obj_file_path
        )
    

