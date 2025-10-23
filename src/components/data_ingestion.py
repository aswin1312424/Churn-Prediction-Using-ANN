import os
import pandas as pd
import numpy as np
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join("data/raw_data","train.csv")
    test_data_path=os.path.join("data/raw_data","test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion")

        df=pd.read_csv("data/raw_data/Churn_Modelling.csv")
        logging.info("Read dataset as Dataframe")

        logging.info("Splitting into train and test")
        train_data,test_data=train_test_split(df,test_size=0.2,random_state=42)

        os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

        train_data.to_csv(self.ingestion_config.train_data_path,index=False)
        test_data.to_csv(self.ingestion_config.test_data_path,index=False)
        logging.info("data ingestion completed")

        return(
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path
        )

  
