import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

class ModelArchitecture:
    def __init__(self):
        pass

    def model_building(self,cleaned_train_data_path):
        train_df=pd.read_csv(cleaned_train_data_path)
        x_train=train_df.iloc[:,:-1]

        model=Sequential([
            Input(shape=(x_train.shape[1],)),
            Dense(64,activation="relu"),
            Dense(32,activation="relu"),
            Dense(1,activation="sigmoid")
        ])

        model.compile(optimizer="Adam",loss="binary_crossentropy",metrics=["accuracy"])

        early_stopping_callback=EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True)

        return(
            model,
            early_stopping_callback
        ) 
    
   