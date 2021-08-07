from os import path, mkdir, remove
import sys
import logging


# function to remove old and add new handlers to logger.

def create_logger(log_dir, file_name):
    # get logger
    logger = logging.getLogger()
    
    # remove previous handlers, if they exist
    if bool(logger.handlers):
        logger.handlers.clear()
    
    # create a log directory, if not exists
    if not path.exists(log_dir):
        mkdir(log_dir)
    
    log_file_path = path.join(log_dir, file_name)
    
    # remove old log file (w/ same name)
    if path.exists(log_file_path):
        remove(log_file_path)
    
    # create a new log file
    f = open(log_file_path, 'w+')
    f.close()

    # configure message to console
    console_logging_format = "%(message)s"

    logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = console_logging_format)

    # create a file handler for output file
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    
    # configure message to log file
    file_logging_format = "[%(levelname)s] %(asctime)s: %(message)s"
    formatter = logging.Formatter(file_logging_format)
    file_handler.setFormatter(formatter)
    
    # add handlers to logger
    logger.addHandler(file_handler)
    
    return logger