import ast
from typing import List

import napari.layers
import napari.types
import numpy as np
from magicgui import magic_factory


@magic_factory
def native_view(
    im: "napari.layers.Image",
) -> List[napari.types.LayerDataTuple]:
    # a little bigger than traditional view, but much faster

    # try:
    #     num_t, num_z, num_y, num_x = im.data.shape
    #     dims = 4
    # except ValueError:
    #     num_t = 1
    #     dims = 3
    #     num_z, num_y, num_x = im.data.shape
    num_t, num_c, num_z, num_y, num_x = im.data.shape
    scan_step_size_px = int(
        im.metadata["snouty_metadata"]["scan_step_size_px"]
    )
    max_deshear_shift = int(np.rint(scan_step_size_px * (num_z - 1)))
    im_desheared = np.zeros(
        (num_t, num_c, num_z, num_y + max_deshear_shift, num_x), im.dtype
    )

    def deshear_channel(ch_num):
        desheard_ch = np.zeros(
            (num_t, 1, num_z, num_y + max_deshear_shift, num_x), im.dtype
        )
        for z in range(num_z):
            deshear_shift = int(np.rint(z * scan_step_size_px))
            desheard_ch[
                :, 0, z, deshear_shift : (deshear_shift + num_y), :
            ] = im.data[:, ch_num, z, :, :]
            # if dims == 4:
            # im_desheared[
            #     :, ch_num, z, deshear_shift:(deshear_shift + num_y), :
            # ] = im.data[:, ch_num, z, :, :]
            # else:
            #     im_desheared[
            #         :, z, deshear_shift : (deshear_shift + num_y), :
            #     ] = im.data[z, :, :]]
        return desheard_ch

    wavelengths = ast.literal_eval(
        im.metadata["snouty_metadata"]["channels_per_slice"]
    )
    layer_tuples = []
    for ch in range(num_c):
        wavelength = int(wavelengths[ch])
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
        ch_desheared = deshear_channel(ch)
        if num_c > 1:
            im_desheared[:, ch, ...] = ch_desheared[:, 0, ...]
            ch_desheared = ch_desheared[:, 0, ...]
        layer_tuples.append(
            (
                ch_desheared,
                {
                    "name": f"{wavelength}-{im.name}",
                    "blending": "additive",
                    "colormap": color,
                    "metadata": im.metadata,
                },
                "image",
            )
        )
    if num_c > 1:
        layer_tuples.insert(
            0,
            (
                im_desheared,
                {
                    "name": f"multichannel-{im.name}",
                    "visible": False,
                    "metadata": im.metadata,
                },
                "image",
            ),
        )
    # todo things are different with 1 ch vs multiple
    # layer_tuple = (im_desheared, {"name": "native-" + im.name}, "image")
    im.visible = False
    # todo remove im
    return layer_tuples
