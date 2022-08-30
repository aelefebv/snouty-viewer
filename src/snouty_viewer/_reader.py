"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/plugins/guides.html?#readers
"""
import glob
import os.path

import dask
import dask.array as da
import tifffile
from napari.utils import progress


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        return None
    return reader_function


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    # handle both a string and a list of strings
    data_path = os.path.join(path, 'data')
    data_tifs = glob.glob(os.path.join(data_path, '*.tif'))
    data_tifs.sort()
    with tifffile.TiffFile(data_tifs[0]) as tif:
        metadata = tif.imagej_metadata
        print(metadata)
        xy_shape = tif.pages[0].shape
        im_dtype = tif.pages[0].dtype
    # volumes_in_file = metadata['frames']
    # num_volumes = volumes_in_file * num_ims
    num_volumes = metadata['frames']
    num_channels = metadata['channels']
    num_z = metadata['slices']

    im_shape = (num_volumes, num_z,) + xy_shape

    @dask.delayed
    def load_tif(im_path, ch):
        with tifffile.TiffFile(im_path) as tif_frame:
            im_frame = tif_frame.asarray()[:, :, ch, ...]
        return im_frame

    def load_channel(all_tifs, ch):
        im_channel = da.zeros(shape=im_shape, dtype=im_dtype)
        for tif_frame in progress(all_tifs):
            im = da.from_delayed(load_tif(tif_frame, ch), shape=im_shape, dtype=im_dtype)
            im_channel = da.append(im_channel, im, axis=0)
        im_channel = im_channel.rechunk({0: 1})
        layer_type = "image"
        layer_name = path.rsplit(os.sep)[-2]
        add_kwargs = {"name": f'ch{ch}_' + layer_name}
        return im_channel, add_kwargs, layer_type

    im_tuples = []
    for channel in range(num_channels):
        im_tuples.append(load_channel(data_tifs, channel))
    return im_tuples
