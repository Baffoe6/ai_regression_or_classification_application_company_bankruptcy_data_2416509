"""
Logging utilities for the bankruptcy prediction project.
"""
import logging
import sys
from typing import Optional
from ..config import Config


def setup_logging(config: Config) -> logging.Logger:
    """Set up logging configuration."""
    # Create logger
    logger = logging.getLogger('bankruptcy_prediction')
    logger.setLevel(getattr(logging, config.log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(config.log_file)
    file_handler.setLevel(getattr(logging, config.log_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get logger instance."""
    if name:
        return logging.getLogger(f'bankruptcy_prediction.{name}')
    return logging.getLogger('bankruptcy_prediction')