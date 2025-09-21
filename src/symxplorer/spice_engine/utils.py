import os
import logging
import shutil

from pathlib import Path


import logging
import os
from datetime import datetime

def setup_loggers(out_logname="default_wrapper", parent_folder:Path=Path(".")) -> logging.Logger:

    # --- Create timestamped log filename ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_logname = Path(f"{parent_folder}/logs/{out_logname}_{timestamp}.log")
    os.makedirs(out_logname.parent, exist_ok=True)

    # --- The wrapper logger ---
    logger = logging.getLogger("wrapper_logger")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s: [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    # Always clear old handlers to avoid duplicates
    logger.handlers.clear()

    # --- Console Handler ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- File Handler ---
    file_handler = logging.FileHandler(out_logname, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("🚀 Logger initialized and ready!")
    logger.info(f"📄 Log file: {os.path.abspath(out_logname)}")

    # --- NEW: Enable debug logging for spicelib ---
    spicelib_logger = logging.getLogger("spicelib")
    spicelib_logger.setLevel(logging.DEBUG)
    spicelib_logger.handlers.clear()  # avoid duplicates
    spicelib_logger.addHandler(console_handler)
    spicelib_logger.addHandler(file_handler)

    logger.info("🔧 spicelib logger set to DEBUG")

    return logger