import os
import sys
from src.logger import logging
from src.exception import CoustomException
import pandas  as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


## initialize the data ingestion configuration

@dataclass
class dataIngestionConfig:
    trainDataPath = os.path.join("artifacts","train.csv")
    testDataPath = os.path.join("artifacts","test.csv")
    rawDataPath = os.path.join("artifacts","raw.csv")
    

## create a data ingestion class

class dataIngestion:
    def __init__(self):
        self.ingestionConfig = dataIngestionConfig()
        
    def initiateDataIngestion(self):
        logging.info("data ingestion start")
        
        try:
            df = pd.read_csv(os.path.join("notebooks/data","gemstone.csv"))
            logging.info("Dataset read as Pandas Dataframe")
            
            os.makedirs(os.path.dirname(self.ingestionConfig.rawDataPath), exist_ok=True)
            df.to_csv(self.ingestionConfig.rawDataPath,index=False)
            
            logging.info("Train Test split ")
            
            trainSet , testSet = train_test_split(df,test_size = 0.30, random_state= 42)
            
            trainSet.to_csv(self.ingestionConfig.trainDataPath, index=False, header = True)
            testSet.to_csv(self.ingestionConfig.testDataPath, index=False, header = True)
            
            logging.info("Ingestion of data is completed")
            
            
            return (
                self.ingestionConfig.trainDataPath, 
                self.ingestionConfig.testDataPath
            )
            
            

            
        except Exception as e :
            logging.error("Error occured in data ingestion config")

   