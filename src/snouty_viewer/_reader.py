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
    metadata_path = os.path.join(path, "metadata", "000000.txt")

    metadata = load_metadata(metadata_path)

    data_tifs = glob.glob(os.path.join(data_path, "*.tif"))
    data_tifs.sort()

    num_buffers = len(data_tifs)

    with tifffile.TiffFile(data_tifs[0]) as tif:
        xy_shape = tif.pages[0].shape
        im_dtype = tif.pages[0].dtype

    vols_per_buffer = int(metadata["volumes_per_buffer"])
    num_volumes = vols_per_buffer * num_buffers
    # print("[INFO] Number of volumes: ", num_volumes)
    channel_str = metadata["channels_per_slice"]
    num_channels = len(channel_str.rsplit(","))
    # print("[INFO] Number of channels: ", num_channels)
    num_z = int(metadata["slices_per_volume"])
    # print("[INFO] Number of slices: ", num_z)

    im_shape = (num_volumes, num_z, xy_shape[0] - 8, xy_shape[1])

    def load_tif(im_path, ch):
        with tifffile.TiffFile(im_path) as tif_frame:
            if ch == -1:
                im_frame = tif_frame.asarray()[..., 8:, :]
            else:
                im_frame = tif_frame.asarray()[..., ch, 8:, :]
        return im_frame

    def load_channel(all_tifs, ch):
        im_channel = np.zeros(shape=im_shape, dtype=im_dtype)
        for idx, tif_frame in enumerate(all_tifs):
            start = vols_per_buffer * idx
            end = start + vols_per_buffer
            im_channel[start:end, ...] = load_tif(tif_frame, ch)
        layer_type = "image"
        layer_name = path.rsplit(os.sep)[-1]
        if ch == -1:
            name = layer_name
        else:
            name = f"ch{ch}_" + layer_name
        add_kwargs = {
            "name": name,
            "metadata": {"path": path, "snouty_metadata": metadata},
        }
        return im_channel, add_kwargs, layer_type

    im_tuples = []
    if num_channels == 1:
        im_channel, add_kwargs, layer_type = load_channel(data_tifs, -1)
        im_tuples.append((im_channel, add_kwargs, layer_type))
    else:
        for channel in range(num_channels):
            im_tuples.append(load_channel(data_tifs, channel))
    return im_tuples
