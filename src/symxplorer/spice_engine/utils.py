import os
import logging
import shutil

from pathlib import Path



def setup_loggers():
    # --- The wrapper logger ---
    logger = logging.getLogger("wrapper_logger")
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logger.propagate = False
    formatter = logging.Formatter(fmt="%(asctime)s - %(name)s: [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(console_handler)
    logger.info("Logger initialized.")

    # --- NEW: Enable debug logging for spicelib ---
    spicelib_logger = logging.getLogger("spicelib")
    spicelib_logger.setLevel(logging.DEBUG)

    # Optionally, attach the same console handler so spicelib logs show up too
    if not spicelib_logger.handlers:
        spicelib_logger.addHandler(console_handler)

    logger.info("spicelib logger set to DEBUG")

    return logger