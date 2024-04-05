import os, sys 
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.exception import CoustomException
from src.logger import logging

from src.utils import saveObject
from src.utils import evaluateModel

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


## model save 

@dataclass
class modelTrainConfig:
    trainModelFilePath = os.path.join("artifacts","model.pkl")
    

class modelTrainer:
    
    def __init__(self):
        self.modelTrainerConfig = modelTrainConfig()
        
    def initiateModelTraining(self, trainArray, testArray):
        try :
            logging.info("Started with data")
            
            logging.info("Split the data into independent and dependent dataset")
            
            X_train, y_train, X_test, y_test = (
                trainArray[:,:-1], trainArray[:,-1], testArray[:,:-1], testArray[:,-1]
            )
            print("data splited into X_train X_test y_train y_test")
            
            models = {
                "linear regression" : LinearRegression(),
                "lasso" : Lasso(),
                "ridge" : Ridge(),
                "elastic net" : ElasticNet()
            }            
            
            modelReport : dict = evaluateModel(X_train, y_train, X_test, y_test, models)
            
            print(modelReport)
            print("\n############################################################################")
            logging.info("\n ###############################################################################")
            logging.info("MODEL REPORT \n")

            logging.info(f"{modelReport}")
            
            bestModelScore = max(sorted(modelReport.values()))
            
            print(f"best model score is : {bestModelScore}")
            
            bestModelName = list(modelReport.keys())[
                list(modelReport.values()).index(bestModelScore)
            ]
            
            print(f"best model name is : {bestModelName}")
                    
            bestModel = models[bestModelName]
            print(f"the final model is {bestModel}")
            
            ## train the best model 
            bestModel.fit(X_train,y_train)
            
            saveObject(
                filePath=self.modelTrainerConfig.trainModelFilePath,
                object= bestModel
                
            )
            
            
        except Exception as e :
            print("Error occure at model traning")
            logging.error("Error Occure while model train")
            CoustomException(e,sys)
            