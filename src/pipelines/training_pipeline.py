import os
import sys
import pandas as pd

from src.logger import logging
from src.exception import CoustomException

from src.components.data_ingestion import dataIngestion
from src.components.data_transformation import dataTransformation
from src.components.model_trainer import modelTrainer


if __name__ == "__main__" :
    
    obj = dataIngestion()   
    trainDataPath, testDataPath = obj.initiateDataIngestion()
    print("traning path is : ",trainDataPath)
    print("testing path is : ", testDataPath)
    
    datatransformation = dataTransformation()
    
    datatransformation.initiateDataTransformation(trainDataPath, testDataPath)
    trainArray, testArray, preprocessorObjectSaveFilePath = datatransformation.initiateDataTransformation(trainDataPath, testDataPath)
    
    print("train array shape", trainArray.shape)
    print("test array shape", testArray.shape)
    
    print("preprocesor save file path is : ", preprocessorObjectSaveFilePath)
    
    modelTrain =modelTrainer() 
    modelTrain.initiateModelTraining(trainArray,testArray)
    
    print("model Trainer is done")
    
        