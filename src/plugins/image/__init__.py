from pathlib import Path
from typing import Union

import h5py
from PIL import Image
from textual.binding import Binding
from textual.reactive import var
from textual.widget import Widget
from textual_imageview.img import ImageView as TextualImageView
from textual_imageview.viewer import ImageViewer as TextualImageViewer

import h5tui.h5
from h5tui import HDF5ItemViewer


class ImageView(TextualImageViewer, can_focus=True):
    """A Widget to display images."""

    DEFAULT_CSS = """
        ImageView {
            height: 1fr;
        }
    """

    image = var(None)

    BINDINGS = [
        Binding("=", "zoom(-1)", "Zoom In", show=False, key_display="+"),
        Binding("-", "zoom(1)", "Zoom Out", show=False, key_display="-"),
        Binding("0", "reset_zoom", "Reset Zoom", show=False, key_display="0"),

        Binding("w,up", "move(0, -1)", "Up", show=True, key_display="W/↑"),
        Binding("s,down", "move(0, +1)", "Down", show=True, key_display="S/↓"),
        Binding("a,left", "move(-1, 0)", "Left", show=True, key_display="A/←"),
        Binding("d,right", "move(+1, 0)", "Right", show=True, key_display="D/→"),
    ]

    def __init__(self, image: Image.Image):
        super().__init__(image)

    def action_zoom(self, delta: int):
        self.zoom(delta)

    def action_reset_zoom(self):
        self.reset_zoom()

    def action_move(self, delta_x: int, delta_y: int):
        self.image.move(delta_x, delta_y)
        self.refresh()

    def update(self, image: Image.Image):
        self.image = TextualImageView(image)
        self.image.set_container_size(*self.container_size)
        self.on_show()

    def zoom(self, delta):
        self.image.zoom(delta)
        self.refresh()

    def reset_zoom(self):
        self.notify("Zoom reset to default.")
        self.on_show()


class ImageViewer(HDF5ItemViewer):
    @staticmethod
    def can_handle(item: Union[h5py.File, h5py.Group, h5py.Dataset]) -> bool:
        if h5tui.h5.is_dataset(item):
            data = h5tui.h5.get_data(item)
            if data.ndim == 2 or data.ndim == 3:
                return True
        return False

    @staticmethod
    def get_widget(item: Union[h5py.File, h5py.Group, h5py.Dataset]) -> Widget:
        # First create an Image from the HDF5 data
        data = h5tui.h5.get_data(item)
        image = Image.fromarray(data)
        return ImageView(image)

    @staticmethod
    def get_id() -> str:
        return "image-view"
