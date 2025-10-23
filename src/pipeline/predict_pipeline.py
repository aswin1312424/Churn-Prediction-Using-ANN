import os
import pandas as pd
import tensorflow as tf
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
        model_path=os.path.join("artifacts","model.h5")

        preprocessor=load_object(preprocessor_path)
        model=tf.keras.models.load_model(model_path)

        scaled_data=preprocessor.transform(features)
        prediction=model.predict(scaled_data)

        return prediction
    
class CustomData:
    def __init__(self,
           credit_score,
           geography,
           gender,
           age,
           tenure,
           balance,
           num_of_products,
           has_cr_card,
           is_active_member,
           estimated_salary):
        
        self.credit_score=credit_score
        self.geography=geography
        self.gender=gender
        self.age=age
        self.tenure=tenure
        self.balance=balance
        self.num_of_products=num_of_products
        self.has_cr_card=has_cr_card
        self.is_active_member=is_active_member
        self.estimated_salary=estimated_salary

    def to_dataframe(self):
        custom_input_dict={
            "CreditScore":[self.credit_score],
            "Geography":[self.geography],
            "Gender":[self.gender],
            "Age":[self.age],
            "Tenure":[self.tenure],
            "Balance":[self.balance],
            "NumOfProducts":[self.num_of_products],
            "HasCrCard":[self.has_cr_card],
            "IsActiveMember":[self.is_active_member],
            "EstimatedSalary":[self.estimated_salary]
        }
        
        return pd.DataFrame(custom_input_dict)
        
if __name__=="__main__":
    obj=CustomData(850,"Germany","Male",49,8,98649.55,1,1,0,119174.88)
    df=obj.to_dataframe()

    pred=PredictPipeline()
    print(pred.predict(df))