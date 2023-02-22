import glob
import os.path

import numpy as np
import tifffile


def napari_get_reader(path):
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        return None
    return reader_function


def load_metadata(metadata_path):
    with open(metadata_path) as f:
        lines = f.readlines()
    metadata_dict = dict()
    for line in lines:
        split = line.split(": ")
        metadata_dict[split[0]] = split[1][:-1]
    return metadata_dict


def reader_function(path):
    data_path = os.path.join(path, "data")
    metadata_dir = os.path.join(path, "metadata")
    metadata_files = glob.glob(os.path.join(metadata_dir, "*.txt"))
    metadata_path = metadata_files[0]

    metadata = load_metadata(metadata_path)

    data_tifs = glob.glob(os.path.join(data_path, "*.tif"))
    data_tifs.sort()

    num_buffers = len(data_tifs)

    with tifffile.TiffFile(data_tifs[0]) as tif:
        xy_shape = tif.pages[0].shape
        im_dtype = tif.pages[0].dtype

    vols_per_buffer = int(metadata["volumes_per_buffer"])
    num_volumes = vols_per_buffer * num_buffers
    channel_str = metadata["channels_per_slice"]
    num_channels = len(channel_str.rsplit(" "))
    num_z = int(metadata["slices_per_volume"])

    im_shape = (num_volumes, num_channels, num_z, xy_shape[0] - 8, xy_shape[1])

    def load_tif(im_path, ch):
        if num_channels == 1:
            im_frame = tifffile.memmap(im_path)[..., 8:, :]
        else:
            im_frame = tifffile.memmap(im_path)[..., ch, 8:, :]
        return im_frame

    def load_channel(all_tifs, ch):
        im_channel = np.zeros(
            shape=(im_shape[0],) + (im_shape[2:]), dtype=im_dtype
        )
        for idx, tif_frame in enumerate(all_tifs):
            start = vols_per_buffer * idx
            end = start + vols_per_buffer
            im_channel[start:end, ...] = load_tif(tif_frame, ch)
        return im_channel

    loaded_im = np.zeros(shape=im_shape, dtype=im_dtype)
    for ch_num in range(num_channels):
        loaded_im[:, ch_num, ...] = load_channel(data_tifs, ch_num)
    px_size = float(metadata["sample_px_um"])
    z_px_size = px_size * float(metadata["voxel_aspect_ratio"])
    scale = (z_px_size, px_size, px_size)
    layer_type = "image"
    name = path.rsplit(os.sep)[-1]
    add_kwargs = {
        "name": name,
        "metadata": {"path": path, "snouty_metadata": metadata},
        "scale": scale,
    }
    im_tuple = [(loaded_im, add_kwargs, layer_type)]
    return im_tuple
