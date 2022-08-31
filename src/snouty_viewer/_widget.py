"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory
import numpy as np

if TYPE_CHECKING:
    import napari


@magic_factory
def native_view(im: "napari.layers.Image") -> "napari.types.LayerDataTuple":  # a little bigger than traditional view, but much faster
    num_t, num_z, num_y, num_x = im.data.shape
    scan_step_size_px = int(im.metadata['snouty_metadata']['scan_step_size_px'])
    max_deshear_shift = int(np.rint(scan_step_size_px * (num_z - 1)))
    im_desheared = np.zeros((num_t, num_z, num_y + max_deshear_shift, num_x), im.dtype)
    for z in range(num_z):
        deshear_shift = int(np.rint(z * scan_step_size_px))
        im_desheared[:, z, deshear_shift:(deshear_shift + num_y), :] = im.data[:, z, :, :]
    layer_tuple = (im_desheared, {"name": "native-" + im.name}, "image")
    return layer_tuple

