import os
import pandas as pd
from dataclasses import dataclass
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_architecture import ModelArchitecture

@dataclass
class TrainingPipelineConfig:
    model_path=os.path.join("artifacts","model.h5")

class TrainingPipeline:
    def __init__(self):
        self.training_config=TrainingPipelineConfig()
    
    def initiate_training_pipeline(self,model,callback,scaled_train_path,scaled_test_path):

        scaled_train_data=pd.read_csv(scaled_train_path)
        scaled_test_data=pd.read_csv(scaled_test_path)

        x_train=scaled_train_data.iloc[:,:-1]
        y_train=scaled_train_data.iloc[:,-1]
        x_test=scaled_test_data.iloc[:,:-1]
        y_test=scaled_test_data.iloc[:,-1]

        model.fit(
            x_train,y_train,validation_data=(x_test,y_test),epochs=50,
            callbacks=[callback]
        )

        os.makedirs(os.path.dirname(self.training_config.model_path),exist_ok=True)

        model.save(self.training_config.model_path)

        return self.training_config.model_path


if __name__=="__main__":
        
    data_ingestion=DataIngestion()
    train_path,test_path=data_ingestion.initiate_data_ingestion()

    data_transformation=DataTransformation()
    cleaned_train_path,cleaned_test_path,_=data_transformation.initiate_data_transformation(train_path,test_path)

    model_architecture=ModelArchitecture()
    model,callback=model_architecture.model_building(cleaned_train_path)

    training_pipeline=TrainingPipeline()
    model_path=training_pipeline.initiate_training_pipeline(model,callback,cleaned_train_path,cleaned_test_path)

        