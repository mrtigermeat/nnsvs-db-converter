import sys
from loguru import logger as logging

global logging

def set_logger(debug):
	if debug:
		log_level = "DEBUG"
	else:
		log_level = "INFO"
	logging.remove(0)
	logging.add(sys.stderr, format="<green>DB CONVERTER</green> | {level} | <level>{message}</level>", level=log_level)