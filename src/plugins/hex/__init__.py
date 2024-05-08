from typing import Union

import h5py
from textual import events
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import RichLog

import h5tui.h5
from h5tui import HDF5ItemViewer


# can_focus_children=False to prevent the RichLog widget from also getting the focus.
# can_focus=True to allow HexView to get focus
class HexView(VerticalScroll, can_focus=False, can_focus_children=True):

    LIMIT = 0x1000
    BINDINGS = [
        Binding('a', 'ascii', "ASCII"),
        Binding('h', 'hex', "HEX"),
        Binding('*', 'reload_full', 'Reload FULL'),
        Binding('/', 'reload_limited', 'Reload LIMIT'),
    ]

    def __init__(self, item, *children: Widget):
        super().__init__(*children)
        self.item = item
        self.limit = self.LIMIT
        self.in_hex = True

    def compose(self) -> ComposeResult:
        yield RichLog(markup=True, auto_scroll=False)

    def on_mount(self):
        rich_log = self.query_one(RichLog)
        self._populate_richlog(rich_log)

    def _on_focus(self, event: events.Focus) -> None:
        ...

    @work(thread=True, exclusive=True)
    async def _populate_richlog(self, rich_log: RichLog):

        line_length = 120
        bytes_per_line = line_length // 3

        if h5tui.h5.is_dataset(self.item):
            rich_log.clear()
            data = h5tui.h5.get_data(self.item)
            data = data.item() if data.shape == () else data.tobytes()

            for idx in range(0, min(len(data), self.limit), bytes_per_line):
                prefix = f"[b]0x{idx:04X}[/]"
                hex_str = " ".join(f"{self._hex(x) if self.in_hex else self._ascii(x)}" for x in data[idx:idx + bytes_per_line])
                # rich_log.write(Text.from_markup(f"{prefix} | {hex_str}"))  # not needed when markup=True
                rich_log.write(f"{prefix} | {hex_str}")

            # rich_log.scroll_end()

    def _hex(self, byte) -> str:
        return f"{byte:02X}" if 32 <= byte <= 126 else f"[dim]{byte:02X}[/]"

    def _ascii(self, byte) -> str:
        return chr(byte) if 32 <= byte <= 126 else "."

    def action_ascii(self):
        self.in_hex = False
        rich_log = self.query_one(RichLog)
        self._populate_richlog(rich_log)

    def action_hex(self):
        self.in_hex = True
        rich_log = self.query_one(RichLog)
        self._populate_richlog(rich_log)

    def action_reload_full(self):
        self.limit = 0xFFFF_FFFF
        rich_log = self.query_one(RichLog)
        rich_log.loading = True
        self._populate_richlog(rich_log)
        rich_log.loading = False

    def action_reload_limited(self):
        self.limit = self.LIMIT
        rich_log = self.query_one(RichLog)
        self._populate_richlog(rich_log)


class HexViewer(HDF5ItemViewer):
    @staticmethod
    def can_handle(item: Union[h5py.File, h5py.Group, h5py.Dataset]) -> bool:
        if h5tui.h5.is_dataset(item):
            data = h5tui.h5.get_data(item)
            return True if data.ndim == 1 else False
        return False

    @staticmethod
    def get_widget(item: Union[h5py.File, h5py.Group, h5py.Dataset]) -> Widget:
        return HexView(item)

    @staticmethod
    def get_id() -> str:
        return "hex-view"
