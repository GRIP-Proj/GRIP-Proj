import os
import torch
import logging
import colorlog


class Logger:
    def __init__(self, output_path=None, multi_worker=False, console=True, level=logging.INFO):
        self.output_path = output_path
        self.multi_worker = multi_worker
        self.level = level

        # Create base logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)

        # Avoid duplicate handlers if Logger() is called multiple times
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        if console:
            formatter = colorlog.ColoredFormatter(
                "%(log_color)s[%(asctime)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
            console_handler = colorlog.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # --- File Handler (optional) ---
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            file_handler = logging.FileHandler(output_path, mode="a")
            file_formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    # --- Wrapper methods for DDP-safe logging ---
    def _is_main_process(self):
        if not self.multi_worker:
            return True
        if not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0

    def debug(self, msg):
        if self._is_main_process():
            self.logger.debug(msg)

    def info(self, msg):
        if self._is_main_process():
            self.logger.info(msg)

    def warning(self, msg):
        if self._is_main_process():
            self.logger.warning(msg)

    def error(self, msg):
        if self._is_main_process():
            self.logger.error(msg)

    def critical(self, msg):
        if self._is_main_process():
            self.logger.critical(msg)
