"""
日志工具模块 —— 统一日志输出格式。
"""

import logging
import sys

# 日志格式
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "coc-keeper") -> logging.Logger:
    """
    创建并返回一个命名 logger，输出到 stdout。
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    return logger


# 模块级 logger，供各模块直接引用
logger = get_logger()
