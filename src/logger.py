import logging
import os
from datetime import datetime

LOG_FILE = F"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

logs_path = os.path.join("logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True)
print("logs path is : ",logs_path)

LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE) 
print("log file path is ", LOG_FILE_PATH)

logging.basicConfig(
    filename = LOG_FILE_PATH,
    format= "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
