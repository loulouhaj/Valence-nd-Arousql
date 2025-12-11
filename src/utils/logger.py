import logging
import os
from datetime import datetime
from config import config

def setup_logging():
    """
    Sets up the python logging module.
    """
    handlers = [logging.StreamHandler()]
    
    if config.LOG_TO_FILE:
        log_file_path = os.path.join(config.LOG_DIR, config.LOG_FILE_NAME)
        handlers.append(logging.FileHandler(log_file_path))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger("TCN_Project")

def get_tensorboard_log_dir(model_name: str = "TCN") -> str:
    """
    Returns a unique log directory for TensorBoard based on timestamp.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(config.LOG_DIR, "fit", f"{model_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

logger = setup_logging()
