import ast
import os
from typing import List, Union

import napari.layers
import napari.types
import numpy as np
from magicgui import magic_factory

from snouty_viewer._writer import write_single_image
from snouty_viewer.im_loader import ImPathInfo, load_full


class PseudoImage:
    def __init__(self, im_tuple: tuple):
        self.data = im_tuple[0]
        self.metadata = im_tuple[1]["metadata"]
        self.name = im_tuple[1]["name"]
        self.dtype = im_tuple[0].dtype


class ImInfo:
    def __init__(self, im):
        (
            self.num_t,
            self.num_c,
            self.num_z,
            self.num_y,
            self.num_x,
        ) = im.data.shape
        self.scan_step_size_px = int(
            im.metadata["snouty_metadata"]["scan_step_size_px"]
        )
        self.max_deshear_shift = int(
            np.rint(self.scan_step_size_px * (self.num_z - 1))
        )
        self.im_desheared = np.zeros(
            (
                self.num_t,
                self.num_c,
                self.num_z,
                self.num_y + self.max_deshear_shift,
                self.num_x,
            ),
            im.dtype,
        )
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

    def _deshear_channel(self, ch_num):
        desheard_ch = np.zeros(
            (
                self.num_t,
                1,
                self.num_z,
                self.num_y + self.max_deshear_shift,
                self.num_x,
            ),
            self.dtype,
        )
        for z in range(self.num_z):
            deshear_shift = int(np.rint(z * self.scan_step_size_px))
            desheard_ch[
                :, 0, z, deshear_shift : (deshear_shift + self.num_y), :
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
            wavelength = int(self.wavelengths[ch])
            if wavelength < 430:
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
                self.im_desheared[:, ch, ...] = ch_desheared[:, 0, ...]
                ch_desheared = ch_desheared[:, 0, ...]
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


@magic_factory(call_button="Deskew and save")
def batch_deskew_and_save(
    path: str, show_deskewed_ims: bool = False, auto_save: bool = True
) -> Union[List[napari.types.LayerDataTuple], None]:
    snouty_dirs = list_subdirectories(path)
    tuple_list = []
    for snouty_dir in snouty_dirs:
        im_path_info = ImPathInfo(snouty_dir)
        loaded_im = PseudoImage(load_full(im_path_info)[0])
        im_info = ImInfo(loaded_im)
        im_info.deshear_all_channels(batch=True, show_multi=show_deskewed_ims)
        if show_deskewed_ims:
            tuple_list.append(im_info.displayed_images[0])
        if auto_save:
            save_path = os.path.join(
                snouty_dir, f"deskewed-{im_info.name}.ome.tif"
            )
            attributes = {"metadata": im_info.metadata}
            write_single_image(save_path, im_info.im_desheared, attributes)

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
