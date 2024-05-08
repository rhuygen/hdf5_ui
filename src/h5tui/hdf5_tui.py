"""
A Textual User Interface for inspection of HDF5 files.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from sys import argv
from typing import List
from typing import Optional
from typing import Union

import h5py
import natsort
import numpy as np
from rich.protocol import is_renderable
from rich.table import Table
from rich.text import Text
from rich.traceback import Traceback
from textual import log
from textual import on
from textual.app import App
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.containers import ScrollableContainer
from textual.containers import Vertical
from textual.reactive import reactive
from textual.reactive import var
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Footer
from textual.widgets import Header
from textual.widgets import Label
from textual.widgets import Static
from textual.widgets import TabPane
from textual.widgets import TabbedContent
from textual.widgets import Tree

from . import h5
from .plugin import load_plugins


NUMERALS = " ⅠⅡⅢⅣⅤⅥ"


class InfoView(Static):
    """A Widget to display generic info."""


class AttributesWidget(Static):
    """A Widget to display attributes in a Table."""


class Views(Vertical):
    """The widget that contains all views in a tabbed content."""
    item = reactive[Union[h5py.File, h5py.Dataset, h5py.Group]](None, init=False)
    added_tabs = var(list)

    def compose(self) -> ComposeResult:
        with TabbedContent():
            with TabPane('item', id="info-view-tab"):
                yield InfoView(id="info-view")

    @on(TabbedContent.TabActivated)
    def report(self, event: TabbedContent.TabActivated):
        self.log(f"TAB {event.pane.id} is now active.")
        # event.pane.focus()

    def watch_item(self):

        tab_content = self.query_one(TabbedContent)
        # Reset the text info view as the active tab, otherwise the newly added tab that gets focus will not be shown
        # for some reason
        tab_content.active = "info-view-tab"
        for pane_id in self.added_tabs:
            tab_content.remove_pane(pane_id)
        self.added_tabs.clear()
        tab_content.refresh()

        if self.item is None:
            return

        item = self.item

        for plugin_name, plugin in self.app.plugins.items():
            if plugin.can_handle(item):
                self.app.log(f"{plugin} can handle {type(self.item)}")
                widget = plugin.get_widget(item)
                tab_id = f"{plugin.get_id()}-tab"
                tab_content.add_pane(TabPane(plugin_name, widget, id=tab_id))
                tab_content.show_tab(tab_id)
                tab_content.active = tab_id
                self.added_tabs.append(tab_id)
                tab_content.refresh(layout=True)
                widget.focus()

                # self.notify(f"{self.container_size = }, {self.content_size = }")

        item_view = self.app.query_one("#info-view", InfoView)

        try:
            if h5.is_dataset(item):
                self.app.log(f"it's a dataset! item={item}, parent={h5.get_parent(item).name}")
                item_view.update(item_to_renderable(item))
            elif h5.is_group(item):
                self.app.log(f"it's a group! item={item}, parent={h5.get_parent(item).name}")
                item_view.update(f"it's a group! item={item.name}")
            elif h5.is_file(item):
                self.app.log(f"it's a file! item={h5.get_filename(item)}")
                item_view.update(f"it's a file! item={h5.get_filename(item)}")
            else:
                item_view.update(f"unknown item: {item.name}")

        except Exception:
            item_view.update(Traceback(theme="github-dark", width=None))


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
            self.app.log(f"{node_data = }")

            try:
                item = node_data['item']
            except KeyError:
                item = node_data

            self.app.log(f"{item = }, {type(item)}")

            views = self.app.query_one(Views)
            views.item = item

        message.stop()

    async def go(self, path: Path) -> None:
        """Loads the structure of the HDF5 into the table of contents."""

        if path is None or not path.exists():
            raise Warning(f"[orange3]Path does not exist:[/] {path!s}")

        try:
            self.hdf5_file = h5.get_file(path, mode='r', locking=False)
        except OSError as exc:
            raise ValueError(f"Unable to load {path!s}") from exc

        self.app.sub_title = str(path)

        self.table_of_contents = populate_data_for_toc(self.hdf5_file, data=[])


class HDF5Browser(App):
    """Textual HDF5 browser app."""

    CSS_PATH = "hdf5_tui.tcss"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("n", "next_file", "Next"),
        ("p", "previous_file", "Previous"),
    ]

    path = var[Optional[Path]](None)
    previous_path = var[Optional[Path]](None)

    def __init__(self):

        super().__init__()

        self.plugins = load_plugins("h5ui.item.view")

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="top-level"):
            with Vertical(id="meta-container"):
                yield TableOfContentsWidget(None, "root", id="tree-view")
                with ScrollableContainer(id="attrs-container"):
                    yield AttributesWidget(id="attrs-view")
            yield Views(id="views-container")
        yield Footer()

    async def on_mount(self) -> None:
        """This function is called after compose and can be used to populate widgets or do further initialisation."""

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
    print(f"{app.plugins = }")

    if len(argv) > 1 and Path(argv[1]).exists():
        app.path = Path(argv[1])

    plugins = load_plugins('h5ui.item.view')
    app.run()


if __name__ == "__main__":
    main()
