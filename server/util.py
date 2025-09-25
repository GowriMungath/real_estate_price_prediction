import json 
import pickle
import pandas as pd
import numpy as np
import os 

__locations=None
__data_columns=None
__model=None

def get_location_names():
    return __locations

def get_estimated_price(location,total_sqft,bhk,bath):
    try:
        loc_index=__data_columns.index(location)
    except:
        loc_index=-1

    x=np.zeros(len(__data_columns))
    x[0]=total_sqft
    x[1]=bath
    x[2]=bhk

    if loc_index>=0:
        x[loc_index]=1

    x_df = pd.DataFrame([x], columns=__data_columns)
    
    return round(__model.predict(x_df)[0],2)

def load_saved_artifacts():
    print("loading saved artifacts... start")
    global __locations
    global __data_columns
    global __model

    script_dir = os.path.dirname(os.path.abspath(__file__))
    columns_path = os.path.join(script_dir, 'artifacts', 'columns.json')
    model_path = os.path.join(script_dir, 'artifacts', 'banglore_home_prices_model.pickle')

    with open(columns_path, 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]
    
    with open(model_path, 'rb') as f:
        __model = pickle.load(f)
        
    print("loading saved artifacts... done")


if __name__=='__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JPs Nagar',1000,3,3))
    print(get_estimated_price('1st Phase JP Nagar',1000,2,2))
    print(get_estimated_price('Kalhalli',1000,2,2))
    print(get_estimated_price('Ejipura',1000,2,2))
