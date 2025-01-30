import os
import logging
from logging.handlers import RotatingFileHandler




logs_dir = 'logs'
os.makedirs(logs_dir, exist_ok=True)

def setup_logging():
    # Log file path
    log_file_path = os.path.join(logs_dir, 'app.log')

    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all log levels

    # Create a file handler
    file_handler = RotatingFileHandler(
        log_file_path, 
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5  # Keep 5 backup files
    )
    file_handler.setLevel(logging.DEBUG)  
    file_handler.setLevel(logging.INFO)  

    # Create a formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    # Optional: Add console handler for immediate feedback during development
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Console only shows INFO and above
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger