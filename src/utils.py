import os
import pickle

def save_object(file_path,obj):
    #This function is to save the object
    
    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    with open(file_path,"wb") as f:
        pickle.dump(obj,f)

def load_object(file_path):
    #This function is to load the object

    with open(file_path,"rb") as f:
        return pickle.load(f)