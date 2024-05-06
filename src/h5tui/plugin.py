__all__ = [
    "entry_points",
    "load_plugins",
]

import importlib.metadata
import logging
from typing import Dict

from typing import Type

from h5tui import HDF5ItemViewer

_LOGGER = logging.getLogger(__name__)


def entry_points(name: str) -> set:
    """Returns the entrypoints for the given package.

    If the package is not installed or can not be found, an empty set is returned.
    """
    try:
        x = importlib.metadata.entry_points()[name]
        return {ep for ep in x}  # use of set here to remove duplicates
    except KeyError:
        return set()


def load_plugins(entry_point: str) -> Dict[str, Type[HDF5ItemViewer]]:
    """Loads all plugins for the given entrypoint and returns a dictionary with the name of the plugin as the key and
    the module or function that was loaded as the value.

    If a plugin can not be loaded, the value will be None.
    """
    eps = {}
    for ep in entry_points(entry_point):
        try:
            eps[ep.name] = ep.load()
        except Exception as exc:
            eps[ep.name] = None
            _LOGGER.error(f"Couldn't load entry point: {exc}")

    return eps
