import logging
import time
import os

STD_LOG_FORMAT = "%(levelname)s\t%(name)s |:|\t%(message)s" # \t%(asctime)s


def setup_logger(log_file_path="./app.log", log_format: str = str(),
                 console_level = logging.DEBUG, file_level = logging.DEBUG):
    """
    Set up logging with a specified log file path and log format.
    
    Parameters:
    - log_file_path (str): Path where the log file should be stored.
    - log_format (str): Logging format string.
    """

    # Ensure logs directory exists
    # os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
    # check if file exists
    if os.path.exists(log_file_path):
        with open(log_file_path, "a") as f:
            f.write("\n\n\n")  # Append blank lines
            # Write date and time
            f.write(f"####################################################################")
            f.write(f"#### STARTING EXPERIMENT AT: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    else: 
        with open(log_file_path, "w") as f:
            # Write date and time
            f.write(f"#### STARTING EXPERIMENT AT: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Default log format if none provided
    if not log_format:
        log_format = STD_LOG_FORMAT

    # Create a formatter
    log_formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(log_formatter)

    # File handler
    file_handler = logging.FileHandler(log_file_path, mode="a")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(log_formatter)

    # Get the root logger
    root_logger = logging.getLogger()
    
    # Prevent adding multiple handlers (avoids duplicate logs)
    if not root_logger.hasHandlers():
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

    logging.getLogger(__name__).info(f"logger initialized with file: {log_file_path}")



