import os
from typing import Type, Union

import numpy as np
import ome_types
import tifffile


class ImPathInfo:
    def __init__(self, path):
        self.path = path
        self.data_path = os.path.join(path, "data")
        self.metadata_dir = os.path.join(path, "metadata")
        metadata_files = [
            f for f in os.listdir(self.metadata_dir) if f.endswith(".txt")
        ]
        self.metadata_files = [
            os.path.join(self.metadata_dir, f) for f in metadata_files
        ]
        self.metadata_path = self.metadata_files[0]

        self.metadata = load_metadata(self.metadata_path)

        data_tifs = [
            f for f in os.listdir(self.data_path) if f.endswith(".tif")
        ]
        self.data_tifs = [os.path.join(self.data_path, f) for f in data_tifs]
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
        self.axes = "TCZYX"


def allocate_memory_return_memmap(
    axes,
    shape,
    snouty_metadata,
    save_path: str,
    dtype: Union[Type, str] = "float",
):
    tifffile.imwrite(
        save_path,
        shape=shape,
        dtype=dtype,
        bigtiff=True,
        metadata={"axes": axes},
    )
    ome_xml = tifffile.tiffcomment(save_path)
    ome = ome_types.from_xml(ome_xml, parser="lxml")
    delay = snouty_metadata["delay_s"]
    if delay is None or delay == "None":
        delay = 0.0
    else:
        delay = float(delay)
    vps = float(snouty_metadata["volumes_per_s"])
    px_size = float(snouty_metadata["sample_px_um"])
    ome.images[0].pixels.physical_size_x = px_size
    ome.images[0].pixels.physical_size_y = px_size
    ome.images[0].pixels.physical_size_z = px_size * float(
        snouty_metadata["voxel_aspect_ratio"]
    )
    ome.images[0].pixels.time_increment = vps + delay
    ome.images[0].description = snouty_metadata["description"]
    # ome.images[0].pixels.type = dtype
    # note: numpy uses 8 bits as smallest, so 'bit' type does nothing for bool.
    ome_xml = ome.to_xml()
    tifffile.tiffcomment(save_path, ome_xml)
    return tifffile.memmap(save_path, mode="r+")


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


def load_full(im_path_info: ImPathInfo, path_out):
    name = im_path_info.path.rsplit(os.sep)[-1]
    save_path = os.path.join(path_out, f"skewed-{name}.ome.tif")
    skewed_memmap = allocate_memory_return_memmap(
        im_path_info.axes,
        im_path_info.im_shape,
        im_path_info.metadata,
        save_path,
        im_path_info.im_dtype,
    )
    for ch_num in range(im_path_info.num_channels):
        if im_path_info.num_channels > 1:
            skewed_memmap[:, ch_num, ...] = load_channel(im_path_info, ch_num)
            # ch_desheared = ch_desheared[:, ...]
        else:
            # self.im_desheared = ch_desheared
            skewed_memmap = load_channel(im_path_info, ch_num)
    px_size = float(im_path_info.metadata["sample_px_um"])
    z_px_size = px_size * float(im_path_info.metadata["voxel_aspect_ratio"])
    scale = (z_px_size, px_size, px_size)
    layer_type = "image"
    add_kwargs = {
        "name": name,
        "metadata": {
            "path": im_path_info.path,
            "snouty_metadata": im_path_info.metadata,
        },
        "scale": scale,
    }
    im_tuple = [(skewed_memmap, add_kwargs, layer_type)]
    return im_tuple


def load_tif(im_path, ch, num_channels=1):
    if num_channels == 1:
        im_frame = tifffile.memmap(im_path, mode="r")[..., 8:, :]
    else:
        im_frame = tifffile.memmap(im_path, mode="r")[..., ch, 8:, :]
    return im_frame


def load_metadata(metadata_path):
    with open(metadata_path) as f:
        lines = f.readlines()
    metadata_dict = dict()
    for line in lines:
        split = line.split(": ")
        metadata_dict[split[0]] = split[1][:-1]
    return metadata_dict
