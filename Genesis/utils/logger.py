"""
Genesis Logger
Logging utility
"""

import logging
import sys
import io
from typing import Optional

# Fix encoding for Windows
if sys.platform == 'win32' and sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def setup_logger(
    name: str = 'Genesis',
    level: str = 'INFO',
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger
    
    Args:
        name: Logger name
        level: Log level
        log_file: Log file path (optional)
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler with UTF-8 encoding
    # Create a UTF-8 encoded stream for Windows
    if sys.platform == 'win32':
        # Use a custom stream that handles encoding properly
        stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
        console_handler = logging.StreamHandler(stream)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
