"""
Utilities for working with ml_collections.ConfigDicts.
"""
import sys
from typing import Type, TypeVar

from absl import flags
from IPython import get_ipython
from ml_collections import config_flags

A = TypeVar("A")


def setup_config(config_class: Type[A]) -> A:
    """Sets up a config dataclass from the command line."""
    config = config_class()
    if not get_ipython():
        config_flag = config_flags.DEFINE_config_dataclass("config", config)
        flags.FLAGS(sys.argv)
        config = config_flag.value
    return config
