import logging
from typing import Optional

def setup_logger(name: str = "optimized_inference", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger for the application"""
    logger = logging.getLogger(name)
    
    # Only configure if handlers haven't been set up
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    
    return logger