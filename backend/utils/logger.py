from __future__ import annotations
import logging, os, sys
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(formatter)
    fh = TimedRotatingFileHandler("logs/backend.log", when="D", interval=1, backupCount=7)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
