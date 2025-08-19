import logging
import os
import sys 
from datetime import datetime 

# Configure logging
LOG_FILE = os.path.join(os.getcwd(), "logs", f"{datetime.now().strftime('%m-%d-%Y-%H-%M-%S')}.log")

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True) if not os.path.exists(os.path.dirname(LOG_FILE)) else None

logging.basicConfig(
    filename=LOG_FILE,
    format='[ %(asctime)s ] - %(lineno)d %(name)s - %(filename)s - %(funcName)s  - %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%m-%d-%Y %H:%M:%S'
)
