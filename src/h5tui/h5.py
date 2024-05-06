"""
H5 - Inspect HDF5 files from the commandline (REPL)

This module uses the h5py module to load and navigate the HDF5. It provides
a number of convenience methods for inspecting the HDF5 file structure and
contents.

Usage: The examples are for PlatoSim3 HDF5 output files

  hfile = h5py.File(<filename>) or myh5 = h5.get_file(filename)

  h5.show_file(myh5)

  myh5.has_groups(hfile["StarPositions"])
  myh5.show_groups(hfile['StarPositions'], recursive=False)

  myh5.has_datasets(hfile["Images"])
  myh5.show_datasets(hfile["Images"])

  myh5.has_attributes(hfile['InputParameters/CCD'])
  myh5.show_attributes(hfile['InputParameters/CCD'])

  colPix = myh5.get_data(out["StarPositions/Exposure000000/colPix"])

  position = myh5.get_attribute_value(out["InputParameters/CCD"], "Position")
"""

import sys

import h5py

INDENT = " " * 4
MAX_LEVEL = 100


def dummy_calc_size_of(x):
    return 0


# Trying to make an option to calc or not the size of objects.
# This approach doesn't work, since the whole dataset is still
# extracted from the HDF5 when choosing no.
calc_size_of = dummy_calc_size_of
calc_size_of = sys.getsizeof


def get_type_id(x) -> str:
    """
    Returns the type of the argument as a single letter.

    * 'D' - Dataset
    * 'G' - Group
    * 'F' - File

    If the type is not one of the options above, ``type(x)`` is returned as a string.

    :return: a character that identifies the type
    """
    if isinstance(x, h5py.File):
        return "F"
    if isinstance(x, h5py.Dataset):
        return "D"
    if isinstance(x, h5py.Group):
        return "G"
    return str(type(x))


def get_type_name(x) -> str:
    """
    Returns the type of the argument as a single letter.

    * 'D' - Dataset
    * 'G' - Group
    * 'F' - File

    If the type is not one of the options above, ``type(x)`` is returned as a string.

    :return: a character that identifies the type
    """
    if isinstance(x, h5py.Dataset):
        return "Dataset"
    if isinstance(x, h5py.Group):
        return "Group"
    if isinstance(x, h5py.File):
        return "File"
    return str(type(x))


# FIXME: replace with egse.bits.humanize_bytes()

def show_size(n) -> str:
    """
    Returns the size ``n`` in human readable form, i.e. as bytes, KB, MB, GB, ...

    :return: human readable size
    """
    _n = n
    if _n < 1024:
        return "{} bytes".format(_n)

    for dim in ['KB', 'MB', 'GB', 'TB', 'PB']:
        _n = _n / 1024.0
        if _n < 1024:
            return "{:.3f} {}".format(_n, dim)

    return "{} bytes".format(n)


def show_file(x):
    """
    Print the top-level groups and attributes in the HDF5 file.

    This will recursively crawl through the HDF5 file and calculate the total size
    of the all the groups and attributes.
    Only the top-level of the HDF5 structure will be printed. The total size of the
    group includes the attributes that belong to that group. Total size of attributes
    is only for top-level attributes.

    """
    if not isinstance(x, h5py.File):
        print("Error: Please pass a HDF5 File argument.")
        return
    __show_groups__(x, recursive=True, max_level=1)
    show_attributes(x)
    return


def get_file(filename, mode='r', locking=None):
    """
    Load the HDF5 file.

    Args:
        filename: the absolute filename of the HDF5 file
        mode: r, r+, w, w-, x, a (see documentation of h5py module) [default='r']
        locking: False to disable locking [default=None]

    Returns:
        a h5py File object.
    """
    return h5py.File(filename, mode=mode, locking=locking)


def get_filename(x):
    return x.file.filename


def is_file(x):
    return get_type_id(x) == 'F'


def get_root(x):
    """Returns the root of this HDF5 file, i.e. `hdf5_file["/"]`."""
    return x.file["/"]


def get_parent(x):
    return x.parent


def get_dataset(x, name: str):
    return x[name]


def get_group(x, name: str):
    return x[name]


def has_attributes(x):
    return len(x.attrs) > 0


def show_attributes(x):
    _size = 0

    for a in x.attrs:
        t_size = calc_size_of(x.attrs[a])
        print("{}: {} ({})".format(a, x.attrs[a], show_size(t_size)))
        _size = _size + t_size

    print("Total size of attributes: {}".format(show_size(_size)))
    pass


def get_attributes(x) -> dict:
    return {
        a: x.attrs[a]
        for a in x.attrs
    }


def get_attribute_value(x, a):
    return x.attrs[a]


def has_groups(x):
    return len([g for g in x.keys() if get_type_id(x[g]) == 'G']) > 0


def is_group(x):
    return get_type_id(x) == 'G'


def groups(x):
    return (x[g] for g in x.keys() if get_type_id(x[g]) == 'G')


def items(x):
    return (x[g] for g in x.keys())


def __show_groups__(x, recursive=True, indent="", verbose=True, max_level=MAX_LEVEL, level=0):
    _indent = indent + INDENT
    _level = level + 1
    _size = 0

    # print ("indent=*{}*, _indent=*{}*".format(indent, _indent))
    if isinstance(x, h5py.Dataset):
        print("Error: you passed in a Dataset. Please provide a HDF5 File or a Group object.")
        return

    for g in x.keys():
        type_id = get_type_id(x[g])

        if has_attributes(x[g]):
            t_size = 0
            for a in x[g].attrs:
                t_size = t_size + calc_size_of(x[g].attrs[a])
            if verbose and max_level > _level:
                print(f"{indent}[A] Total size: {show_size(t_size)}")
            _size = _size + t_size

        if type_id == 'D':
            t_size = calc_size_of(x[g][...])
            if verbose and max_level > level:
                print(f"{indent}[{type_id}] {g} ({show_size(t_size)})")
            _size = _size + t_size
        else:
            if verbose and max_level > level:
                print(f"{indent}[{type_id}] {g}")
            if recursive:
                _size = _size + __show_groups__(x[g], indent=_indent, verbose=verbose, max_level=max_level, level=_level)

    if max_level > level:
        print(f"{indent}Total size of Group = {show_size(_size)}")

    return _size


def show_groups(x, recursive=True, verbose=True, max_level=MAX_LEVEL):
    """
    :param x: HDF5 group object
    :param recursive: crawl over all groups recursively if True
    :param verbose:
    :param max_level:
    """
    __show_groups__(x, recursive=recursive, verbose=verbose, max_level=max_level)


def has_datasets(x):
    return len([g for g in x.keys() if get_type_id(x[g]) == 'D']) > 0


def is_dataset(x):
    return get_type_id(x) == 'D'


def datasets(x):
    """
    Returns a generator for all datasets in 'x'.

    :param x:
    :return:
    """
    return (x[g] for g in x.keys() if get_type_id(x[g]) == 'D')


def show_datasets(x):
    _size = 0
    for key in x.keys():
        type_id = get_type_id(x[key])

        if type_id == 'D':
            t_size = calc_size_of(x[key][...])
            print("[{}] {} ({})".format(type_id, key, show_size(t_size)))
            _size = _size + t_size

    print("Total size of datasets in this group is {}".format(show_size(_size)))


def show_dataset(x):
    if not isinstance(x, h5py.Dataset):
        print("Error: Please provide a HDF5 Dataset as input.")
        return
    print("Shape: {}, Type: {}, Size: {}".format(x.shape, x.dtype, show_size(calc_size_of(x[...]))))
    pass


def get_data(x):
    if not isinstance(x, h5py.Dataset):
        print("Error: Please provide a HDF5 Dataset as input.")
        return None

    return x[...]


__all__ = [
    "show_file",
    "get_type_id",
    "has_groups",
    "show_groups",
    "has_datasets",
    "show_datasets",
    "show_dataset",
    "get_data",
    "has_attributes",
    "show_attributes",
    "get_attribute_value"
]
