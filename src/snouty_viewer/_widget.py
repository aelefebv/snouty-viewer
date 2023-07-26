import ast
import os
from typing import List, Union

import napari.layers
import napari.types
import numpy as np
from magicgui import magic_factory

from scripts.split_positions import process_directory
from snouty_viewer.im_loader import (
    ImPathInfo,
    allocate_memory_return_memmap,
    load_full,
)


class PseudoImage:
    def __init__(self, im_tuple: tuple):
        self.data = im_tuple[0]
        self.metadata = im_tuple[1]["metadata"]
        self.name = im_tuple[1]["name"]
        self.dtype = im_tuple[0].dtype


class ImInfo:
    def __init__(self, im, im_shape=None):
        if im_shape is None:
            im_shape = im.shape
        (
            self.num_t,
            self.num_c,
            self.num_z,
            self.num_y,
            self.num_x,
        ) = im_shape
        self.scan_step_size_px = int(
            im.metadata["snouty_metadata"]["scan_step_size_px"]
        )
        self.max_deshear_shift = int(
            np.rint(self.scan_step_size_px * (self.num_z - 1))
        )
        self.im_desheared_shape = (
            self.num_t,
            self.num_c,
            self.num_z,
            self.num_y + self.max_deshear_shift,
            self.num_x,
        )
        self.im_desheared = None
        # get desheared image memmap
        # self.im_desheared = np.zeros(
        #     (
        #         self.num_t,
        #         self.num_c,
        #         self.num_z,
        #         self.num_y + self.max_deshear_shift,
        #         self.num_x,
        #     ),
        #     im.dtype,
        # )
        self.px_size = float(im.metadata["snouty_metadata"]["sample_px_um"])
        self.z_px_size = self.px_size * float(
            im.metadata["snouty_metadata"]["voxel_aspect_ratio"]
        )
        self.scale = (self.z_px_size, self.px_size, self.px_size)
        self.dtype = im.dtype
        self.metadata = im.metadata
        self.data = im.data
        self.wavelengths = ast.literal_eval(
            im.metadata["snouty_metadata"]["channels_per_slice"]
        )
        self.name = im.name
        self.displayed_images = []

    # def _allocate_memory(self,
    #                      path_im: str,
    #                      dtype: Union[Type, str] = 'float', data = None,
    #                      shape: tuple = None,
    #                      description: str = 'No description.'):
    #     axes = 'TCZYX'
    #     if shape is None:
    #
    #
    #     if data is None:
    #         assert shape is not None
    #         tifffile.imwrite(
    #             path_im, shape=shape, dtype=dtype, bigtiff=True,
    #             metadata={"axes": axes}
    #         )
    #     else:
    #         tifffile.imwrite(
    #             path_im, data, bigtiff=True, metadata={"axes": axes}
    #         )
    #     ome_xml = tifffile.tiffcomment(path_im)
    #     ome = ome_types.from_xml(ome_xml, parser="lxml")
    #     ome.images[0].pixels.physical_size_x = self.dim_sizes['X']
    #     ome.images[0].pixels.physical_size_y = self.dim_sizes['Y']
    #     ome.images[0].pixels.physical_size_z = self.dim_sizes['Z']
    #     ome.images[0].pixels.time_increment = self.dim_sizes['T']
    #     ome.images[0].description = description
    #     ome.images[0].pixels.type = dtype
    # note: numpy uses 8 bits as smallest, so 'bit' type does nothing for bool.
    #     ome_xml = ome.to_xml()
    #     tifffile.tiffcomment(path_im, ome_xml)

    def _deshear_channel(self, ch_num):
        if self.num_c == 1:
            desheard_ch = self.im_desheared
            for z in range(self.num_z):
                deshear_shift = int(np.rint(z * self.scan_step_size_px))
                desheard_ch[
                    :, z, deshear_shift : (deshear_shift + self.num_y), :
                ] = self.data[:, z, :, :]
        else:
            desheard_ch = self.im_desheared[:, ch_num, :, :, :]
            for z in range(self.num_z):
                deshear_shift = int(np.rint(z * self.scan_step_size_px))
                desheard_ch[
                    :, z, deshear_shift : (deshear_shift + self.num_y), :
                ] = self.data[:, ch_num, z, :, :]
        return desheard_ch

    def _display_image(
        self, im, wavelength=0, color="gray", multichannel=False
    ):
        if multichannel:
            self.displayed_images.insert(
                0,
                (
                    im,
                    {
                        "name": f"multichannel-{self.name}",
                        "visible": False,
                        "metadata": self.metadata,
                        "scale": self.scale,
                    },
                    "image",
                ),
            )
        else:
            self.displayed_images.append(
                (
                    im,
                    {
                        "name": f"{wavelength}-{self.name}",
                        "blending": "additive",
                        "colormap": color,
                        "metadata": self.metadata,
                        "scale": self.scale,
                    },
                    "image",
                )
            )
        return None

    def deshear_all_channels(self, batch=False, show_multi=False):
        for ch in range(self.num_c):
            try:
                wavelength = int(self.wavelengths[ch])
            except ValueError:
                wavelength = 0
            if wavelength == 0:
                color = "gray"
            elif wavelength < 430:
                color = "bop purple"
            elif wavelength < 480:
                color = "blue"
            elif wavelength < 500:
                color = "cyan"
            elif wavelength < 570:
                color = "green"
            elif wavelength < 590:
                color = "yellow"
            elif wavelength < 630:
                color = "bop orange"
            elif wavelength < 700:
                color = "red"
            else:
                color = "magenta"
            ch_desheared = self._deshear_channel(ch)
            if self.num_c > 1:
                self.im_desheared[:, ch, ...] = ch_desheared[:, ...]
                ch_desheared = ch_desheared[:, ...]
            else:
                self.im_desheared = ch_desheared
            if not batch:
                self._display_image(ch_desheared, wavelength, color)

        if (self.num_c > 1 and not batch) or show_multi:
            self._display_image(self.im_desheared, multichannel=True)
        return None


def list_subdirectories(path):
    items = os.listdir(path)
    subdirectories = [
        item for item in items if os.path.isdir(os.path.join(path, item))
    ]
    # checks subdirectories to see if it contains a data and metadata folder
    # (indication of Snouty folder)
    snouty_subdirectories = [
        os.path.join(path, subdirectory)
        for subdirectory in subdirectories
        if all(
            dirs in os.listdir(os.path.join(path, subdirectory))
            for dirs in ["data", "metadata"]
        )
    ]
    return snouty_subdirectories


@magic_factory(call_button="Position extraction")
def position_extraction(
    path_in: str,
    path_out: str = "",
) -> None:
    if path_out == "":
        path_out = path_in
    process_directory(path_in, path_out)
    return None


@magic_factory(call_button="Deskew and save")
def batch_deskew_and_save(
    path_in: str,
    path_out: str = "",
    show_deskewed_ims: bool = False,
    auto_save: bool = True,
) -> Union[List[napari.types.LayerDataTuple], None]:
    snouty_dirs = list_subdirectories(path_in)
    tuple_list = []
    for snouty_dir in snouty_dirs:
        if path_out == "":
            path_out = snouty_dir
        im_path_info = ImPathInfo(snouty_dir)
        skewed_memmap = PseudoImage(load_full(im_path_info, path_out)[0])
        im_info = ImInfo(skewed_memmap, im_path_info.im_shape)
        name = im_path_info.path.rsplit(os.sep)[-1]
        # skewed_memmap_path = os.path.join(path_out, f"skewed-{name}.ome.tif")
        save_path = os.path.join(path_out, f"deskewed-{name}.ome.tif")
        deskewed_memmap = allocate_memory_return_memmap(
            "TCZYX",
            im_info.im_desheared_shape,
            im_path_info.metadata,
            save_path,
            im_info.dtype,
        )
        im_info.im_desheared = deskewed_memmap
        im_info.deshear_all_channels(batch=True, show_multi=show_deskewed_ims)
        if show_deskewed_ims:
            tuple_list.append(im_info.displayed_images[0])
        # if auto_save:
        #     save_path = os.path.join(
        #         path_out, f"deskewed-{im_info.name}.ome.tif"
        #     )
        #     attributes = {"metadata": im_info.metadata}
        #     write_single_image(save_path, im_info.im_desheared, attributes)
        # os.remove(skewed_memmap_path)
    if show_deskewed_ims:
        return tuple_list
    return None


@magic_factory(call_button="Deskew")
def native_view(
    im: "napari.layers.Image",
) -> List[napari.types.LayerDataTuple]:
    im_info = ImInfo(im)
    im_info.deshear_all_channels(batch=False, show_multi=False)
    im.visible = False
    return im_info.displayed_images
