import glob
import os.path

import dask
import dask.array as da
import tifffile
from napari.utils import progress
import numpy as np


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


def load_metadata(metadata_path):
    with open(metadata_path) as f:
        lines = f.readlines()
    metadata_dict = dict()
    for line in lines:
        split = line.split(': ')
        metadata_dict[split[0]] = split[1][:-1]
    return metadata_dict


def reader_function(path):

    # handle both a string and a list of strings
    data_path = os.path.join(path, 'data')
    metadata_path = os.path.join(path, 'metadata', '000000.txt')

    metadata = load_metadata(metadata_path)
    data_tifs = glob.glob(os.path.join(data_path, '*.tif'))

    data_tifs.sort()
    num_buffers = len(data_tifs)
    with tifffile.TiffFile(data_tifs[0]) as tif:
        # metadata = tif.imagej_metadata
        xy_shape = tif.pages[0].shape
        im_dtype = tif.pages[0].dtype
    # volumes_in_file = metadata['frames']
    # num_volumes = volumes_in_file * num_ims
    num_volumes = int(metadata['volumes_per_buffer']) * num_buffers
    print("[INFO] Number of volumes: ", num_volumes)
    channel_str = metadata['channels_per_slice']
    num_channels = len(channel_str.rsplit(','))
    print("[INFO] Number of channels: ", num_channels)
    # num_channels = metadata['channels']
    num_z = int(metadata['slices_per_volume'])
    print("[INFO] Number of slices: ", num_z)

    im_shape = (num_volumes, num_z,) + xy_shape

    # @dask.delayed
    def load_tif(im_path, ch):
        with tifffile.TiffFile(im_path) as tif_frame:
            if ch == -1:
                im_frame = tif_frame.asarray()
            else:
                im_frame = tif_frame.asarray()[:, :, ch, ...]
        return im_frame

    def load_channel(all_tifs, ch):
        # im_channel = da.zeros(shape=im_shape, dtype=im_dtype)
        im_channel = np.zeros(shape=im_shape, dtype=im_dtype)
        for idx, tif_frame in enumerate(all_tifs):
            # im_channel[idx, ...] = da.from_delayed(load_tif(tif_frame, ch), shape=im_shape[1:], dtype=im_dtype)
            im_channel[idx, ...] = load_tif(tif_frame, ch)
            # im_channel = da.append(im_channel, im, axis=0)
            # print(im.shape)
        # im_channel = im_channel.rechunk({0: 1})
        layer_type = "image"
        layer_name = path.rsplit(os.sep)[-1]
        if ch == -1:
            name = layer_name
        else:
            name = f'ch{ch}_' + layer_name
        add_kwargs = {"name": name}
        return im_channel, add_kwargs, layer_type

    im_tuples = []
    if num_channels == 1:
        im_tuples.append(load_channel(data_tifs, -1))
    else:
        for channel in range(num_channels):
            im_tuples.append(load_channel(data_tifs, channel))
    return im_tuples
