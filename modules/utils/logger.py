import logging
from typing import Optional
import os
from modules.utils.paths import WEBUI_DIR


def get_logger(name: Optional[str] = None):
    if name is None:
        name = "Voxtral-WebUI"
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler for app.log
        log_file_path = os.path.join(WEBUI_DIR, "app.log")
        file_handler = logging.FileHandler(log_file_path, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger