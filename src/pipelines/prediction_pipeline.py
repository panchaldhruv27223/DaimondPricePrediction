import sys, os
from src.exception import CoustomException
from src.logger import logging
from src.utils import loadObject

import pandas as pd 
import numpy as np

class predictionPipeline:
    
    def __init__(self) :
        pass
    
    def prediction(self,features):
        try :
            
            preprocessorPath = os.path.join("artifacts","preprocessor.pkl")
            modelPath = os.path.join("artifacts","model.pkl")
            
            preprocessor = loadObject(preprocessorPath)
            logging.info("the preprocessor is loadded")
            
            model = loadObject(modelPath)
            logging.info("the model is loadded")
            
            dataScaled = preprocessor.transform(features)
            logging.info("the data is transformed")
            
            predictedOutput = model.predict(dataScaled)
            logging.info("the output is generated")
            
            return predictedOutput
        
            
        except Exception as e :
            print("Exception occure while generated the output")
            CoustomException(e,sys)
            
            
class customData:
    
    def __init__(self,
                 carat : float,
                 depth : float,
                 table : float,
                 x : float,
                 y : float,
                 z : float,
                 cut : str,
                 color : str,
                 clarity :str) :
        
        self.carat = carat,
        self.depth = depth,
        self.table = table,
        self.x = x,
        self.y = y,
        self.z = z,
        self.cut = cut,
        self.color = color,
        self.clarity : clarity
        
        
    def getDataAsDataFrame(self):
        try :
            
            customDataInputDict = {
                "carat" : [self.carat],
                "depth" : [self.depth],
                "table" : [self.table],
                "x" : [self.x],
                "y" : [self.y ],
                "z" : [self.z],
                "cut"   : [self.cut],
                "color" : [self.color],
                "clarity" : [self.clarity]
            }
            
            df = pd.DataFrame(customDataInputDict)
            print("df is : ", df)
            logging.info("Data frame is gathered")
            
            return df       
        
        except Exception as e :
            logging.error(e)
            CoustomException(e,sys)
        