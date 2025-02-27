import sys
from loguru import logger as logging

def set_logger(debug: bool = False):
	if debug:
		log_level = "DEBUG"
	else:
		log_level = "INFO"
	logging.add(sys.stderr, format="{time} | {level} | {message}", level=log_level)