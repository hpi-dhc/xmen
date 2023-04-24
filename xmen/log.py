import logging
import os
from pathlib import Path
from rich.logging import RichHandler

"""
Configures a logger for the current Python script with a rich shell handler for logging.
"""

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# configure shell log
shell_handler = RichHandler()
shell_handler.setLevel(logging.INFO)
fmt_shell = "%(message)s"
shell_formatter = logging.Formatter(fmt_shell)
shell_handler.setFormatter(shell_formatter)
logger.addHandler(shell_handler)
