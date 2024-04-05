import sys, os
import pickle
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CoustomException

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def saveObject(filePath,object):
    try :
        
        dir_path = os.path.dirname(filePath)
        print("from utils : dir path is : ",dir_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(filePath,"wb") as fileObject:
            pickle.dump(obj=object,file=fileObject)
        
        print("from utils : the preprocessor saved.")
        
        
    except Exception as e :
        logging.error("Error occure during the save the preprocessor object")
        CoustomException(e,sys)
        
        
        
        
def evaluateModel(X_train, y_train, X_test, y_test, models):
    try :
        report = {}
        
        print("total number of models is : ",len(models))
        print(X_train,y_train)
        
        for i in range(len(models)):
            
            model = list(models.values())[i]
            print("model name is : ",model)
            model.fit(X_train,y_train)
            
            y_predict = model.predict(X_test)
            
            testModelScore = r2_score(y_test,y_predict)
            
            report[list(models.keys())[i]] = testModelScore
            
        return report
                 
        
    except Exception as e :
        logging.error("Error occure while model evaluate")
        
        
        
def loadObject(filePath):
    
    try :
        with open(filePath,"rb") as fileObj:
            return pickle.load(fileObj)

    except Exception as e :
        logging.error("Exception occure while loading the model")
        CoustomException(e,sys)