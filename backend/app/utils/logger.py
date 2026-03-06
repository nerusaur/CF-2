"""
ChildFocus - Logger Utility
backend/app/utils/logger.py
"""

import logging
import os
from datetime import datetime

# Configure logging
LOG_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level   = logging.INFO,
    format  = "[%(asctime)s] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "childfocus.log")),
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger("childfocus")


def log_classification(video_id: str, label: str, mode: str = "full"):
    """Log a classification event."""
    logger.info(f"CLASSIFY [{mode.upper()}] video_id={video_id} label={label}")


def log_error(context: str, error: Exception):
    """Log an error with context."""
    logger.error(f"ERROR [{context}] {type(error).__name__}: {error}")