import os
import sys

from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    try:
        logging.info("Saving object to file: {}".format(file_path))
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        logging.info("Directory created at: {}".format(dir_path))
        
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.info("Error in save_object function")
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        logging.info("Evaluating models")
        model_report = {}
        
        for i in range(len(models)):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            logging.info(f"Training model: {model_name}")
            model.fit(X_train, y_train) # Train model
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            model_report[model_name] = train_model_score, test_model_score
            logging.info(f"Model: {model_name}, R2 Score: {test_model_score}")
        
        return model_report
    
    except Exception as e:
        logging.info("Error in evaluate_model function")
        raise CustomException(e, sys)