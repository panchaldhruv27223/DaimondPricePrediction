import os, sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from dataclasses import dataclass

from src.logger import logging
from src.exception import CoustomException
from src.utils import saveObject

## Data transformation config

@dataclass
class dataTransformationConfig:
    preprocessorObjFilePath = os.path.join("artifacts","preprocessor.pkl")


## data Transformation class    

class dataTransformation:
    
    def __init__(self) :    
        self.dataTransformationConfig = dataTransformationConfig()
        
        
    def getDataTransformationObject(self):
        try :
            logging.info("Data Transformation Initiated")
            
            ## define the columns in categorcal featureas and numerical featires 
            categoricalFeatures = ['cut', 'color', 'clarity']
            numericalFeatures =  ['carat', 'depth', 'table', 'x', 'y', 'z']
            
            
            ## Define the coustom category for each categorica features
            cutCategory = ["Fair","Good","Very Good","Premium","Ideal"]
            colorCategory = ["D","E","F","G","H","I","J"]
            clarityCategory = ["I1", "SI2", "SI1", "VS2","VS1","VVS2","VVS1","IF"]



            logging.info("Pipeline initiated")
            ## Numerical pipeline
            numericalPipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                    ]
            )

            ## Categorical Pipeline

            categoricalPipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("ordinalEncoder",OrdinalEncoder(categories=[cutCategory,colorCategory,clarityCategory] ) ),
                    ("standerScaler",StandardScaler() ) 
                ]
            )
        
            preprocessor = ColumnTransformer(
                [ 
                ("numerical pipeline", numericalPipeline, numericalFeatures),
                ("categorical pipeline", categoricalPipeline, categoricalFeatures) 
                ]
            )
            
            logging.info("pipeline is completed")
            
            return preprocessor
        
        except Exception as e:
            logging.error("Error in data transfomation")
            raise CoustomException(e,sys)
        
        
    def initiateDataTransformation(self,trainDataPath,testDataPath):
        try :
            
            trainDf = pd.read_csv(trainDataPath)
            testDf = pd.read_csv(testDataPath)
            
            logging.info("Read train test data complete")
            logging.info(f"train data head is : {trainDf.head(5)}")
            logging.info(f"test data head is : {testDf.head(5)}")
            
            print(1)
            preprocessorObject = self.getDataTransformationObject()
            print(2)
            
            targetColumns = "price"
            dropColumns = [targetColumns, "id"]
            
            print("target columns is :", targetColumns)
            print("columns we have to drop is : ",dropColumns)
            
            ## Features divide into independent and dependent features
            
            inputFeatureTrainDf = trainDf.drop(columns=dropColumns,axis=1)
            targetFeatureTrainDf = trainDf[targetColumns]
            
            inputFeatureTestDf = testDf.drop(columns=dropColumns, axis=1)
            targteFeatureTestDf = testDf[targetColumns]

            # print(inputFeatureTrainDf.head())
            # print(inputFeatureTestDf.head())
            
            
            ## Apply the transformation
            
            inputFeatureTrainArray = preprocessorObject.fit_transform(inputFeatureTrainDf)
            
            inputFeatureTestArray = preprocessorObject.transform(inputFeatureTestDf)
            
            trainArray = np.c_[inputFeatureTrainArray,np.array(targetFeatureTrainDf)]
            testArray = np.c_[inputFeatureTestArray,np.array(targteFeatureTestDf)]
            # print(trainArray)
            
            
            
            logging.info("Applying the preprocessor to both train and test input features.")
            
            
            saveObject(
                filePath=self.dataTransformationConfig.preprocessorObjFilePath,
                object = preprocessorObject
            )
            
            logging.info("Preprocessor pickle is created and saved")
            
            
            return(trainArray, testArray, self.dataTransformationConfig.preprocessorObjFilePath)
            
            
            
        except Exception as e :
            logging.error("Error occure in Data initiaion")
            CoustomException(e,sys)
            