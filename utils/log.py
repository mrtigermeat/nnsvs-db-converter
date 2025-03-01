import sys
from loguru import logger as logging

global logging

def set_logger(debug: bool = False):
	if debug:
		log_level = "DEBUG"
	else:
		log_level = "INFO"
	logging.remove(0)
	logging.add(sys.stderr, format="<green>DB CONVERTER</green> | \t{level}\t |  <level>{message}</level>", level=log_level)

if __name__ == "__main__":
	set_logger(debug=True)

	logging.info("INFO Test")
	logging.warning("WARNING Test")
	logging.debug("DEBUG Test")