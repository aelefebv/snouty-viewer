import os.path

from snouty_viewer.im_loader import ImPathInfo, load_full


def napari_get_reader(path):
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        return None
    return reader_function


def reader_function(path):
    im_path_info = ImPathInfo(path)
    im_tuple = load_full(im_path_info)
    return im_tuple
