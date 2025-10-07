import logging, os
from logging.handlers import TimedRotatingFileHandler

def setup_logger(name: str) -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    fh = TimedRotatingFileHandler("logs/app.log", when="midnight", backupCount=7, encoding="utf-8")
    fh.setLevel("DEBUG")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
