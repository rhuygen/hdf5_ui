"""
Code browser example.

Run with:

    python code_browser.py PATH
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from sys import argv
from typing import List
from typing import Optional

import h5py
import natsort
import numpy as np
from PIL import Image
from rich.protocol import is_renderable
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.traceback import Traceback
from textual import log
from textual import on
from textual.app import App
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.containers import ScrollableContainer
from textual.containers import Vertical
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.reactive import var
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Footer
from textual.widgets import Header
from textual.widgets import Static
from textual.widgets import Tree
from textual_imageview.img import ImageView
from textual_imageview.viewer import ImageViewer

import egse.fee.ffee
import egse.fee.nfee
from egse import h5
from egse.exceptions import InternalError
from egse.reg import RegisterMap
from egse.setup import Setup
from egse.setup import get_setup
from egse.setup import load_setup
from egse.spw import DataPacket
from egse.spw import HousekeepingPacket
from egse.spw import SpaceWirePacket

NUMERALS = " ⅠⅡⅢⅣⅤⅥ"


class ImageWidget(ImageViewer, can_focus_children=True):

    def __init__(self, image: Image.Image | None, id=None):
        if image is None:
            image = Image.new('L', (2295, 4540), 0)
        super().__init__(image)
        self.id = id

    def update(self, image: Image.Image):
        self.image = ImageView(image)
        self.image.set_container_size(*self.container_size)
        self.on_show()

    def zoom(self, delta):
        self.image.zoom(delta)
        self.refresh()

    def reset_zoom(self):
        self.on_show()

    def zoom_to_brightest_pixel(self, zoom: int = 0):

        image_viewer = self
        image_view = self.image

        # Find the image coordinates of the brightest pixel

        array = np.array(image_view.image)

        max_value = np.max(array)  # we might want to report this in the future
        y, x = np.unravel_index(np.argmax(array), array.shape)

        # Before positioning, first zoom out to make the calculations more accurate since the zoomed size
        # will be closer to the image size.

        image_view.set_zoom(zoom, (0, 0))

        # Size of the original image and of the zoomed image
        img_w, img_h = image_view.size
        img_z_w, img_z_h = image_view.zoomed_size

        # Location of the brightest pixel in the image space
        img_x = int(img_z_w / img_w * x)
        img_y = int(img_z_h / img_h * y)

        # Define the canvas width and height in image space (instead of character space)
        image_viewer_w, image_viewer_h = image_viewer.size.width, image_viewer.size.height
        canvas_w, canvas_h = image_viewer_w, image_viewer_h * 2

        # Put the brightest pixel in the center of the canvas
        new_orig_x = (-canvas_w // 2) + img_x
        new_orig_y = (-canvas_h // 2) + img_y

        image_view.origin_position = (new_orig_x, new_orig_y)

        self.refresh()


class ItemWidget(Static):
    """A Widget to display generic info."""


class AttributesWidget(Static):
    """A Widget to display attributes in a Table."""


class HousekeepingWidget(Widget):

    packet = reactive(None)
    camera_type = var(None)

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static(id="hk-view-header")
            yield Static(id="hk-view-data")

    def watch_packet(self, packet: HousekeepingPacket):
        if self.camera_type == "N-FEE":
            data = egse.fee.nfee.HousekeepingData(packet.data)
        elif self.camera_type == "F-FEE":
            type_ = {2: "DEB", 3: "AEB1"}[packet.type.packet_type]
            data = egse.fee.ffee.HousekeepingData(type_, packet.data)
        else:
            data = "[red]ERROR: Unknown camera type, cannot visualise housekeeping packet.[/]"
        header_view = self.query_one("#hk-view-header", Static)
        header_view.update(str(packet))
        data_view = self.query_one("#hk-view-data", Static)
        data_view.update(data)


class TableOfContentsWidget(Widget, can_focus_children=True):
    """Displays a table of contents for the HDF5 file."""

    table_of_contents = reactive[Optional[List[tuple]]](None, init=False)
    """Underlying data to populate the table of contents widget."""

    def __init__(
            self,
            filename: Path | str | None,
            name: str | None = None,
            id: str | None = None,
            classes: str | None = None,
            disabled: bool = False,
    ) -> None:
        """Initialize a table of contents.

        Args:
            filename: the full path to the HDF5 file.
            name: The name of the widget.
            id: The ID of the widget in the DOM.
            classes: The CSS classes for the widget.
            disabled: Whether the widget is disabled or not.
        """
        if filename is not None:
            self.hdf5_file = h5.get_file(filename, mode='r', locking=False)
        else:
            self.hdf5_file = None

        self.camera_type = None

        super().__init__(name=name, id=id, classes=classes, disabled=disabled)

    def compose(self) -> ComposeResult:
        tree: Tree = Tree("HDF5", id="content-tree")
        tree.show_root = True
        tree.show_guides = True
        tree.guide_depth = 4
        tree.auto_expand = True
        yield tree

    def watch_table_of_contents(self, table_of_contents: List[tuple]) -> None:
        """Triggered when the table of contents changes."""
        self.rebuild_table_of_contents(table_of_contents)

    def rebuild_table_of_contents(self, table_of_contents: List[tuple]) -> None:
        """Rebuilds the tree representation of the table of contents data.

        Args:
            table_of_contents: Table of contents.
        """

        # self.app.log(table_of_contents)

        tree = self.query_one(Tree)
        tree.clear()
        root = tree.root

        for (level, name, size, type_name), expandable, item in table_of_contents:

            # self.app.log(f"{level=}, {name=}, {size=}, {type_name=}")

            if level == 0:
                root.data = item
                continue

            node = root
            for _ in range(level - 1):
                if node._children:
                    node = node._children[-1]
                    node.allow_expand = True
                else:
                    node = node.add(NUMERALS[level], expand=False)
            node_type = h5.get_type_id(item)
            node_label = Text.assemble((f"{node_type} ", "dim"), name, f" ({size=})" if node_type == 'G' else '')
            node.add_leaf(node_label, {"item": item})

        root.expand()

    @on(Tree.NodeHighlighted)
    def show_attributes(self, event: Tree.NodeHighlighted):
        # log.info(f"{event = }, {event.node = }, {event.node.data = }")

        if h5.is_file(event.node.data):
            item = event.node.data
        elif isinstance(event.node.data, dict) and 'item' in event.node.data:
            item = event.node.data['item']
        else:
            return

        attr_view = self.app.query_one("#attrs-view", AttributesWidget)
        if h5.has_attributes(item):
            table = Table(title="Attributes")
            table.add_column("name")
            table.add_column("value")

            for k, v in h5.get_attributes(item).items():
                table.add_row(k, str(v))

            attr_view.update(table)
        else:
            attr_view.update("")

    @on(Tree.NodeSelected)
    async def show_item(self, message: Tree.NodeSelected) -> None:
        node_data = message.node.data
        if node_data is not None:
            # await self._post_message(
            #     Markdown.TableOfContentsSelected(self.markdown, node_data["item"])
            # )
            self.app.log(f"{node_data = }")

            if h5.is_file(node_data):
                return

            item = node_data['item']
            self.app.log(f"{item = }")

            item_view = self.app.query_one("#item-view", ItemWidget)
            image_view = self.app.query_one("#image-view", ImageWidget)
            hk_view = self.app.query_one("#hk-view", HousekeepingWidget)

            item_view.display = False
            image_view.display = False
            hk_view.display = False

            try:
                if h5.is_dataset(item):
                    self.app.log(f"it's a dataset! item={item}, parent={h5.get_parent(item).name}")
                    if "register" in item.name:
                        data = RegisterMap(self.camera_type, memory_map=h5.get_data(item), setup=self.app.setup)
                        item_view.display = True
                        item_view.update(data)
                    elif "setup" in item.name:
                        setup_id = item_to_int(item)
                        setup = get_setup(setup_id)
                        item_view.display = True
                        item_view.update(setup)
                    elif "timecode" in item.name:
                        self.app.log(f"it's a timecode!")
                        timecode = h5.get_data(item)
                        self.app.log(f"Visualising {item.name}: {timecode!s}")
                        item_view.display = True
                        item_view.update(f"{timecode = !s}")
                    elif "hk_data" in item.name:
                        if self.camera_type == "N-FEE":
                            data = egse.fee.nfee.HousekeepingData(h5.get_data(item))
                        else:
                            data = egse.fee.ffee.HousekeepingData(h5.get_data(item))
                        item_view.display = True
                        item_view.update(data)
                    elif "hk" in item.name:
                        hk_view.display = True
                        packet = HousekeepingPacket(h5.get_data(item))
                        hk_view.camera_type = self.camera_type
                        hk_view.packet = packet
                        self.app.query_one("#item-container").refresh()
                    elif "/data" in item.name:
                        data = h5.get_data(item)
                        item_view.display = True
                        if len(data) > 10 and DataPacket.is_data_packet(data):
                            packet = DataPacket.create_packet(data)
                            item_view.update(str(packet))
                        else:
                            item_view.update(str(data))
                    elif "/commands" in h5.get_parent(item).name:
                        data = h5.get_data(item)
                        if isinstance(data, np.ndarray):
                            data = data.item()
                        if isinstance(data, bytes):
                            data = data.decode()
                        self.log(f"command {data = }")
                        data = stringify_function_call(data)
                        item_view.display = True
                        item_view.update(Syntax(data, "python", theme="monokai", line_numbers=True))
                    else:
                        item_view.display = True
                        item_view.update(item_to_renderable(item))
                elif h5.is_group(item):
                    if "data" in item:
                        image = self.handle_image_data(item)
                        image_view.display = True
                        image_view.update(image)
                    elif "commands" in item:
                        ...  # datasets are handled above

                    else:
                        pass
                        # item_view.update(f"unknown group item: {item.name}")
                else:
                    item_view.display = True
                    item_view.update(f"unknown item: {item.name}")

            except Exception:
                item_view.update(Traceback(theme="github-dark", width=None))

        message.stop()

    def handle_image_data(self, item) -> Image.Image:
        data_item = h5.get_group(item, "data")

        if "v_start" not in data_item.attrs:
            v_start = 0
            v_end = 2257
            h_end = 2295
            # return Image.new('L', (2295, 4540), 0)
        else:
            v_start = data_item.attrs["v_start"]
            v_end = data_item.attrs["v_end"]
            h_end = data_item.attrs["h_end"]

        return self.create_image(data_item, v_start, v_end, h_end)

    def create_image(self, hdf_item: h5py.Group, v_start: int, v_end: int, h_end: int) -> Image.Image:

        data_group = hdf_item

        ccd_sides = self.app.setup.camera.fee.ccd_sides.enum

        if self.camera_type == 'N-FEE':
            ccd_side = ccd_sides.LEFT_SIDE
            image_creator = ImageCreatorFullSizeN(v_start, v_end, h_end, ccd_sides)
        else:
            ccd_side = h5.get_attribute_value(data_group, "ccd_side")
            image_creator = ImageCreatorFullSizeF(v_start, v_end, h_end, ccd_side, ccd_sides)

        for data_count in sorted(data_group, key=int):
            data = h5.get_data(data_group[data_count])

            if not isinstance(data, np.ndarray):
                raise InternalError('HDF5 data group should only contain numpy arrays.')

            data_packet = SpaceWirePacket.create_packet(data)

            image_creator.add_data(data_packet)

        if self.camera_type == 'N-FEE':
            image = image_creator.get_image(ccd_side).astype(float)
        else:
            image = image_creator.get_image().astype(float)

        image = Image.fromarray(image).rotate(90, expand=True)
        return image

    async def go(self, path: Path) -> None:
        """Loads the structure of the HDF5 into the table of contents."""

        if path is None or not path.exists():
            raise Warning(f"[orange3]Path does not exist:[/] {path!s}")

        try:
            self.hdf5_file = h5.get_file(path, mode='r', locking=False)
        except OSError as exc:
            raise ValueError(f"Unable to load {path!s}") from exc

        self.app.sub_title = str(path)

        if "fee" in self.hdf5_file:
            self.camera_type = h5.get_attribute_value(self.hdf5_file['fee'], 'type')
        else:
            self.camera_type = "N-FEE"

        if "setup" in self.hdf5_file:
            self.app.setup = get_setup(int(h5.get_data(self.hdf5_file['setup'])))
        else:
            self.app.setup = load_setup()

        self.table_of_contents = populate_data_for_toc(self.hdf5_file, data=[])


class HDF5Browser(App):
    """Textual HDF5 browser app."""

    CSS_PATH = "hdf5_tui.tcss"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("n", "next_file", "Next"),
        ("p", "previous_file", "Previous"),

        Binding("=", "zoom(-1)", "Zoom In", show=False, key_display="+"),
        Binding("-", "zoom(1)", "Zoom Out", show=False, key_display="-"),
        Binding("0", "reset_zoom", "Reset Zoom", show=False, key_display="0"),
        Binding("b", "zoom_to_brightest_pixel(3)", "Zoom to brightest pixel", show=False, key_display="B"),

        Binding("w,up", "move(0, -1)", "Up", show=True, key_display="W/↑"),
        Binding("s,down", "move(0, +1)", "Down", show=True, key_display="S/↓"),
        Binding("a,left", "move(-1, 0)", "Left", show=True, key_display="A/←"),
        Binding("d,right", "move(+1, 0)", "Right", show=True, key_display="D/→"),

    ]

    path = var[Optional[Path]](None)
    previous_path = var[Optional[Path]](None)
    setup = var[Optional[Setup]](None)

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="top-level"):
            with Vertical(id="meta-container"):
                yield TableOfContentsWidget(None, "root", id="tree-view")
                with ScrollableContainer(id="attrs-container"):
                    yield AttributesWidget(id="attrs-view")
            with VerticalScroll(id="item-container"):
                yield ItemWidget(id="item-view")
                yield ImageWidget(image=None, id="image-view")
                yield HousekeepingWidget(id="hk-view")
        yield Footer()

    async def on_mount(self) -> None:
        toc = self.query_one(TableOfContentsWidget)
        toc.focus()

        try:
            await toc.go(self.path)
        except FileNotFoundError:
            await self.push_screen(WarningModal(message=f"Unable to load {self.path!r}"))
            self.path = self.previous_path
            self.previous_path = None
        except Warning as exc:
            await self.app.push_screen(WarningModal(str(exc)))
            self.path = self.previous_path
            self.previous_path = None
        except Exception as exc:
            await self.app.push_screen(TracebackModal(exc))
            self.path = self.previous_path
            self.previous_path = None

    async def action_next_file(self) -> None:
        if self.path is None:
            return

        try:
            path = find_next_file(self.path)
        except Exception as exc:
            await self.push_screen(TracebackModal(exc))
            return

        if path is not None:
            self.previous_path = self.path
            self.path = path
            self.log(f"{path = }")
            await self.on_mount()

    async def action_previous_file(self) -> None:
        if self.path is None:
            return

        try:
            path = find_previous_file(self.path)
        except Exception as exc:
            await self.push_screen(TracebackModal(exc))
            return

        if path is not None:
            self.previous_path = self.path
            self.path = path
            self.log(f"{path = }")
            await self.on_mount()

    def action_zoom(self, delta: int):
        image_viewer = self.query_one("#image-view", ImageWidget)
        image_viewer.zoom(delta)

    def action_reset_zoom(self):
        image_viewer = self.query_one("#image-view", ImageWidget)
        image_viewer.reset_zoom()

    def action_move(self, delta_x: int, delta_y: int):
        image_viewer = self.query_one("#image-view", ImageWidget)

        image_viewer.image.move(delta_x, delta_y)
        image_viewer.refresh()
        self.refresh()

    def action_zoom_to_brightest_pixel(self, zoom: int):
        image_viewer = self.query_one("#image-view", ImageWidget)
        image_viewer.zoom_to_brightest_pixel(zoom=zoom)


class ImageCreatorFullSizeN:
    """Create a full size image for the N-CAM."""

    # This version creates the full CCD images and fills the data when it comes in through data packets

    MAX_NR_LINES = 4540
    MAX_NR_COLUMNS = 2295

    def __init__(self, v_start: int, v_end: int, h_end: int, n_fee_side):
        # LOGGER.debug(f"{v_start=}, {v_end=}, {h_end=}, {id(self)=}")

        self.n_fee_side = n_fee_side

        self.nr_lines = v_end - v_start + 1
        self.nr_columns = h_end + 1
        self.index_left = self.index_right = v_start * self.MAX_NR_COLUMNS
        # self.image_E = np.empty((self.MAX_NR_LINES * self.MAX_NR_COLUMNS,), dtype=np.uint16)
        # self.image_F = np.empty((self.MAX_NR_LINES * self.MAX_NR_COLUMNS,), dtype=np.uint16)
        self.image_left = np.full((self.MAX_NR_LINES * self.MAX_NR_COLUMNS,), fill_value=np.nan, dtype=np.uint16)
        self.image_right = np.full((self.MAX_NR_LINES * self.MAX_NR_COLUMNS,), fill_value=np.nan, dtype=np.uint16)

    def add_data(self, data_packet: DataPacket):
        data = data_packet.data_as_ndarray
        data_length = len(data)
        if data_packet.type.ccd_side == self.n_fee_side.LEFT_SIDE:
            self.image_left[self.index_left:self.index_left + data_length] = data
            self.index_left += data_length
        else:
            self.image_right[self.index_right:self.index_right + data_length] = data
            self.index_right += data_length

    def get_image(self, side: int):
        if side == self.n_fee_side.LEFT_SIDE:
            return self.image_left.reshape(self.MAX_NR_LINES, self.MAX_NR_COLUMNS).T
        else:
            return np.fliplr(self.image_right.reshape(self.MAX_NR_LINES, self.MAX_NR_COLUMNS)).T


class ImageCreatorFullSizeF:
    """
    Create an image from incoming data packets for the F-CAM.

    Args:
        v_start (int): starting row of the image (base zero)
        v_end (int): last rows of this image
        h_end (int): last pixel index for readout register
        ccd_side (int): the CCD side (0, 1) for this image (used to decide if image needs to be flipped)
        ccd_sides (enum): camera dependent enumeration for the CCD side
    """
    # This version creates the full CCD images and fills the data when it comes in through data packets

    MAX_NR_LINES = 2255 + 15  # ccd + overscan rows
    MAX_NR_COLUMNS = 2295  # prescan + ccd + overscan

    def __init__(self, v_start: int, v_end: int, h_end: int, ccd_side: int, ccd_sides):
        log.info(f"{v_start=}, {v_end=}, {h_end=}, {id(self)=}, {ccd_side=}, {type(ccd_sides)=}")

        self.ccd_side = ccd_side
        self.ccd_sides = ccd_sides
        self.nr_lines = v_end - v_start + 1
        self.nr_columns = h_end + 1
        self.index = v_start * self.MAX_NR_COLUMNS
        self.image = np.full((self.MAX_NR_LINES * self.MAX_NR_COLUMNS,), fill_value=np.nan, dtype=np.uint16)

    def add_data(self, data_packet: DataPacket):
        data = data_packet.data_as_ndarray
        data_length = len(data)

        self.image[self.index:self.index + data_length] = data
        self.index += data_length

    def get_image(self):
        if self.ccd_side == self.ccd_sides.LEFT_SIDE:
            return self.image.reshape(self.MAX_NR_LINES, self.MAX_NR_COLUMNS).T
        else:
            return np.fliplr(self.image.reshape(self.MAX_NR_LINES, self.MAX_NR_COLUMNS)).T


class WarningModal(ModalScreen[None]):

    DEFAULT_CSS = """
    WarningModal {
        align: center middle;
    }    
    """

    BINDINGS = [
        ("escape", "dismiss", "help"),
    ]

    def __init__(self, message):
        super().__init__()
        self.message: str = message

    def compose(self) -> ComposeResult:
        yield WarningDialog(f"[orange3]WARNING — [/]{self.message}")


class WarningDialog(ScrollableContainer):

    DEFAULT_CSS = """
    WarningDialog {
        width: 50%;
        height: 50%;
        border: solid cornflowerblue;
    }
    """

    def __init__(self, message):
        super().__init__()
        self.message: str = message

    def on_mount(self):
        self.scroll_end(animate=False)

    def compose(self) -> ComposeResult:
        yield Static(self.message)


class TracebackModal(ModalScreen[None]):

    DEFAULT_CSS = """
    TracebackModal {
        align: center middle;
    }    
    """

    BINDINGS = [
        ("escape", "dismiss", "help"),
    ]

    def __init__(self, exc):
        super().__init__()
        self.exc: Exception = exc

    def compose(self) -> ComposeResult:
        yield TracebackDialog(self.exc)


class TracebackDialog(ScrollableContainer):

    DEFAULT_CSS = """
    TracebackDialog {
        width: 100;
        height: 50%;
        border: solid cornflowerblue;
    }
    """

    def __init__(self, exc):
        super().__init__()
        self.exc: Exception = exc

    def on_mount(self):
        self.scroll_end(animate=False)

    def compose(self) -> ComposeResult:
        yield Static(Traceback.from_exception(self.exc.__class__, self.exc, self.exc.__traceback__), id="traceback")


def populate_data_for_toc(group, data, level=1):

    # log.info(f"{group = }, {group.name = }, {level = }")

    if h5.is_file(group):
        data.append(([0, "/", 1, "File"], True, group))

    if len(group) < 2500:
        items = natsort.natsorted(group.items(), key=lambda x: x[0])
    else:
        items = group.items()

    # log.info(f"{group = }, {items = }")

    for name, item in items:

        if h5.is_group(item):
            # log.info(f"handling group {name} going into next level ({level} -> {level + 1})...")

            name = item.name.split('/', maxsplit=level)[level]
            size = len(item)
            type_name = h5.get_type_name(item)
            data.append(([level, name, size, type_name], True, item))

            data = populate_data_for_toc(item, data, level + 1)

        elif h5.is_dataset(item):
            # log.info(f"handling dataset {name} at {level = }")

            data.append(([level, name, 1, h5.get_type_name(item)], False, item))

        else:
            log.info(f"{name} is not a group nor a dataset")

    # log.info(f"{data = }")

    return data


def stringify_function_call(function_info: str) -> str:
    def quote(value):
        return f'"{value}"' if isinstance(value, str) else value

    # First match the function_info string. This string is auto generated and should have the same format always.

    pattern = r"^(.+), args=\[(.*)\], kwargs=(\{.*\})$"
    match = re.match(pattern, function_info)
    name = match.group(1)
    args = match.group(2)
    kwargs = match.group(3)

    # log.info(f"{name = }, {args = }, {kwargs = }")

    result = name
    result += "("

    if args:
        result += args

    if kwargs:
        result += ", " if args else ""
        result += ", ".join([f"{k}={quote(v)}" for k, v in eval(kwargs).items()])

    result += ")"

    return result


def item_to_int(item):
    """Return a single value item as an integer."""
    data = h5.get_data(item)
    if isinstance(data, np.ndarray):
        data = data.item()
    if isinstance(data, bytes):
        data = data.decode()
    return int(data)


def item_to_renderable(item):
    """Return a single value item as a Rich renderable."""
    data = h5.get_data(item)
    if isinstance(data, np.ndarray):
        data = data.item()
    if isinstance(data, bytes):
        data = data.decode()
    if not is_renderable(data):
        data = str(data)
    return data


def find_next_file(current_file: Path | str) -> Path | None:

    current_file = Path(current_file).expanduser()
    filename = current_file.name
    directory = current_file.parent

    if not directory.exists():
        raise ValueError(f"Directory doesn't exist: {directory!s}")

    files = sorted([f for f in os.listdir(directory) if f.endswith('.hdf5')])

    try:
        idx = files.index(filename)
    except ValueError:
        raise

    if idx + 1 < len(files):
        return directory / files[idx+1]
    else:
        return None


def find_previous_file(current_file: Path | str) -> Path | None:

    current_file = Path(current_file).expanduser()
    filename = current_file.name
    directory = current_file.parent

    if not directory.exists():
        return None

    files = sorted([f for f in os.listdir(directory) if f.endswith('.hdf5')])

    try:
        idx = files.index(filename)
    except ValueError:
        return None

    if idx > 0:
        return directory / files[idx-1]
    else:
        return None


def main():
    app = HDF5Browser()

    print(f"{argv = }")

    if len(argv) > 1 and Path(argv[1]).exists():
        app.path = Path(argv[1])

    app.setup = load_setup()
    app.run()


if __name__ == "__main__":
    main()
