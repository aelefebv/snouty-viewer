import glob
import os

import numpy as np
import tifffile


class ImPathInfo:
    def __init__(self, path):
        self.path = path
        self.data_path = os.path.join(path, "data")
        self.metadata_dir = os.path.join(path, "metadata")
        self.metadata_files = glob.glob(
            os.path.join(self.metadata_dir, "*.txt")
        )
        self.metadata_path = self.metadata_files[0]

        self.metadata = load_metadata(self.metadata_path)

        self.data_tifs = glob.glob(os.path.join(self.data_path, "*.tif"))
        self.data_tifs.sort()

        self.num_buffers = len(self.data_tifs)

        with tifffile.TiffFile(self.data_tifs[0]) as tif:
            self.xy_shape = tif.pages[0].shape
            self.im_dtype = tif.pages[0].dtype

        self.vols_per_buffer = int(self.metadata["volumes_per_buffer"])
        self.num_volumes = self.vols_per_buffer * self.num_buffers
        self.channel_str = self.metadata["channels_per_slice"]
        self.num_channels = len(self.channel_str.rsplit(" "))
        self.num_z = int(self.metadata["slices_per_volume"])

        self.im_shape = (
            self.num_volumes,
            self.num_channels,
            self.num_z,
            self.xy_shape[0] - 8,
            self.xy_shape[1],
        )


def load_channel(im_path_info, ch):
    im_channel = np.zeros(
        shape=(im_path_info.im_shape[0],) + (im_path_info.im_shape[2:]),
        dtype=im_path_info.im_dtype,
    )
    for idx, tif_frame in enumerate(im_path_info.data_tifs):
        start = im_path_info.vols_per_buffer * idx
        end = start + im_path_info.vols_per_buffer
        im_channel[start:end, ...] = load_tif(
            tif_frame, ch, im_path_info.num_channels
        )
    return im_channel


def load_full(im_path_info):
    loaded_im = np.zeros(
        shape=im_path_info.im_shape, dtype=im_path_info.im_dtype
    )
    for ch_num in range(im_path_info.num_channels):
        loaded_im[:, ch_num, ...] = load_channel(im_path_info, ch_num)
    px_size = float(im_path_info.metadata["sample_px_um"])
    z_px_size = px_size * float(im_path_info.metadata["voxel_aspect_ratio"])
    scale = (z_px_size, px_size, px_size)
    layer_type = "image"
    name = im_path_info.path.rsplit(os.sep)[-1]
    add_kwargs = {
        "name": name,
        "metadata": {
            "path": im_path_info.path,
            "snouty_metadata": im_path_info.metadata,
        },
        "scale": scale,
    }
    im_tuple = [(loaded_im, add_kwargs, layer_type)]
    return im_tuple


def load_tif(im_path, ch, num_channels=1):
    if num_channels == 1:
        im_frame = tifffile.memmap(im_path)[..., 8:, :]
    else:
        im_frame = tifffile.memmap(im_path)[..., ch, 8:, :]
    return im_frame


def load_metadata(metadata_path):
    with open(metadata_path) as f:
        lines = f.readlines()
    metadata_dict = dict()
    for line in lines:
        split = line.split(": ")
        metadata_dict[split[0]] = split[1][:-1]
    return metadata_dict
