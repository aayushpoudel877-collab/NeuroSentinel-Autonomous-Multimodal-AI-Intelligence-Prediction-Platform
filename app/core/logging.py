import logging
from app.core.config import settings
def configure_logging():
    logging.basicConfig(level=getattr(logging,settings.log_level.upper(),logging.INFO),format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
def get_logger(name): return logging.getLogger(name)
