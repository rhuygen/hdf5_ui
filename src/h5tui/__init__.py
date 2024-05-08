from abc import ABC
from abc import abstractmethod
from typing import Union

import h5py
from textual.widget import Widget


class HDF5ItemViewer(ABC):
    @staticmethod
    @abstractmethod
    def can_handle(item: Union[h5py.File, h5py.Group, h5py.Dataset]) -> bool:
        ...

    @staticmethod
    @abstractmethod
    def get_widget(item: Union[h5py.File, h5py.Group, h5py.Dataset]) -> Widget:
        ...

    @staticmethod
    @abstractmethod
    def get_id() -> str:
        ...
