import importlib.metadata

__version__ = importlib.metadata.version(__package__ or __name__)

from xmen.kb import load_kb
from xmen.confhelper import load_config
