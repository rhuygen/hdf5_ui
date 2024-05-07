# A Terminal User Interface for inspecting HDF5 files

This App is developed with [Textual](https://www.textualize.io).

![](https://raw.githubusercontent.com/rhuygen/hdf5_ui/develop/docs/images/h5tui_ex01.png)
![](https://raw.githubusercontent.com/rhuygen/hdf5_ui/develop/docs/images/h5tui_ex03.png)

# Feature

This TUI is currently in beta and lacks a lot of features. Following is a list implemented features:

- Press 'p' and 'n' to load the 'previous' or 'next' HDF5 file. This is useful when inspecting a range of HDF5 files at the same location and you want to quickly browse through them.
- Content of the HDF5 file is reflected in the tree structure at the left. 
- If a node has attributes attached, they will be displayed at the bottom-left when the node is highlighted.
- Plugins:
  - The App loads all known plugins at startup and uses these plugins to visualize the information they can handle.
  - Two example plugins have been included: HexViewer, and ImageViewer.

# License

This App is distributed under the MIT license.
