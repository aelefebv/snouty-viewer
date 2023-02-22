import ast
from typing import List

import napari.layers
import napari.types
import numpy as np
from magicgui import magic_factory


@magic_factory(call_button="Deskew")
def native_view(
    im: "napari.layers.Image",
) -> List[napari.types.LayerDataTuple]:
    num_t, num_c, num_z, num_y, num_x = im.data.shape
    scan_step_size_px = int(
        im.metadata["snouty_metadata"]["scan_step_size_px"]
    )
    max_deshear_shift = int(np.rint(scan_step_size_px * (num_z - 1)))
    im_desheared = np.zeros(
        (num_t, num_c, num_z, num_y + max_deshear_shift, num_x), im.dtype
    )
    px_size = float(im.metadata["snouty_metadata"]["sample_px_um"])
    z_px_size = px_size * float(
        im.metadata["snouty_metadata"]["voxel_aspect_ratio"]
    )
    scale = (z_px_size, px_size, px_size)

    def deshear_channel(ch_num):
        desheard_ch = np.zeros(
            (num_t, 1, num_z, num_y + max_deshear_shift, num_x), im.dtype
        )
        for z in range(num_z):
            deshear_shift = int(np.rint(z * scan_step_size_px))
            desheard_ch[
                :, 0, z, deshear_shift : (deshear_shift + num_y), :
            ] = im.data[:, ch_num, z, :, :]
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
                    "scale": scale,
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
                    "scale": scale,
                },
                "image",
            ),
        )
    im.visible = False
    return layer_tuples
