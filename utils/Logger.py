# Copyright (c) 2025 Yeelysia. All rights reserved.

import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.setup_logger()
    
    def setup_logger(self):
        self.logger = logging.getLogger('logger')
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(fmt="%(asctime)s [ %(filename)s ]  %(lineno)d行 | [ %(levelname)s ] | [%(message)s]", datefmt="%Y/%m/%d/%X")
        sh = logging.StreamHandler()

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        log_name_appendex = "{0}.txt".format(datetime.now().strftime('%Y-%m-%d %H-%M-%S')) # type: ignore
        filename = os.path.join(self.log_dir, log_name_appendex)
        fh = logging.FileHandler(filename, encoding="utf-8")

        self.logger.addHandler(sh)
        sh.setFormatter(formatter)
        self.logger.addHandler(fh)
        fh.setFormatter(formatter)
        
        
    def debug(self, message: str):
        self.logger.debug(message)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def critical(self, message: str):
        self.logger.critical(message)

logger = Logger(log_dir='logs')