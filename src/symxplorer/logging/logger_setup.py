import os
import logging
import sys

from pathlib import Path

import logging
import os
from datetime import datetime

def setup_loggers(out_logname="SymXplorer", parent_folder:Path=Path(".")) -> logging.Logger:

    # --- Create timestamped log filename ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_logname = Path(f"{parent_folder}/logs/{out_logname}_{timestamp}.log")
    os.makedirs(out_logname.parent, exist_ok=True)

    # --- The wrapper logger ---
    logger = logging.getLogger("SymXplorer")
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
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- File Handler ---
    file_handler = logging.FileHandler(out_logname, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("🚀 Logger initialized and ready!")
    logger.info(f"📄 Log file: {os.path.abspath(out_logname)}")



    # --- Configure logging for spicelib ---
    # spicelib_logger = setup_spicelib_logging(file_handler)
    spicelib_logger = logging.getLogger("spicelib")
    spicelib_logger.setLevel(logging.CRITICAL)
    
    logger.info(f"🔧 spicelib logger set to {spicelib_logger.getEffectiveLevel()}")

    return logger

def setup_spicelib_logging(file_handler: logging.FileHandler) -> logging.Logger:
    # Get the top-level spicelib logger
    logger = logging.getLogger("spicelib")
    logger.setLevel(logging.INFO)  # Master level must be low enough to let handlers filter

    # --- Console handler (only CRITICAL)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.CRITICAL)
    console_formatter = logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)

    # Clear old handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Attach handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Ensure all children inherit this setup
    for name, temp_logger in logging.Logger.manager.loggerDict.items():
        if name.startswith("spicelib."):
            logging.getLogger(name).setLevel(logging.INFO)
            if isinstance(temp_logger, logging.Logger):
                temp_logger.addHandler(console_handler)
                temp_logger.addHandler(file_handler)

    return logger