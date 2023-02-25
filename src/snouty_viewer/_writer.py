from typing import Any, Dict, List

import numpy as np
import tifffile

# from ome_types.model import OME, Image, Pixels, Plane, Instrument, Channel


def create_ome(layer_data, metadata):
    omexml = tifffile.OmeXml()
    t, z, c, y, x = layer_data.shape
    omexml.addimage(
        dtype=layer_data.dtype,
        shape=layer_data.shape,
        storedshape=(z * t * c, 1, 1, y, x, 1),
        axes="TZCYX",
        PhysicalSizeX=metadata["PhysicalSizeX"],
        PhysicalSizeY=metadata["PhysicalSizeY"],
        PhysicalSizeZ=metadata["PhysicalSizeZ"],
        TimeIncrement=metadata["finterval"],
        SamplesPerPixel=3,
    )
    return omexml.tostring()


def write_single_image(
    path: str, layer_data: Any, attributes: Dict
) -> List[str]:
    # savetype = "imagej"
    savetype = "ome"
    if len(layer_data.shape) == 4:
        layer_data = np.expand_dims(layer_data, axis=1)
    layer_data = np.swapaxes(layer_data, 1, 2)
    px_size = float(attributes["metadata"]["snouty_metadata"]["sample_px_um"])
    z_px_size = px_size * float(
        attributes["metadata"]["snouty_metadata"]["voxel_aspect_ratio"]
    )
    vps = float(attributes["metadata"]["snouty_metadata"]["volumes_per_s"])
    spb = float(attributes["metadata"]["snouty_metadata"]["buffer_time_s"])
    imagej_metadata = {
        "axes": "TZCYX",
        "PhysicalSizeX": px_size,
        "PhysicalSizeY": px_size,
        "PhysicalSizeZ": z_px_size,
        "PhysicalSizeXUnit": "um",
        "PhysicalSizeYUnit": "um",
        "VolumesPerSecond": vps,
        "SecondsPerBuffer": spb,
        "spacing": z_px_size,
        "unit": "um",
        "finterval": 1 / vps + spb,
        "fps": 1 / (1 / vps + spb),
    }
    num_t, num_z, num_c, num_y, num_x = layer_data.shape
    # ome_metadata = {
    #     "axes": "TZCYX",
    #     "PhysicalSizeX": px_size,
    #     "PhysicalSizeY": px_size,
    #     "PhysicalSizeZ": z_px_size,
    #     "PhysicalSizeXUnit": "um",
    #     "PhysicalSizeYUnit": "um",
    #     "PhysicalSizeZUnit": "um",
    #     "TimeIncrement": 1 / vps + spb,
    #     "SamplesPerPixel": 3,
    #     "SizeC": num_c,
    #     "SizeT": num_t,
    #     "SizeZ": num_z,
    #     "SizeY": num_y,
    #     "SizeX": num_x,
    #     "spacing": z_px_size,
    #     "unit": "um",
    #     "finterval": 1 / vps + spb,
    #     "fps": 1 / (1 / vps + spb),
    # }
    ome = create_ome(layer_data, imagej_metadata)
    # for k, v in attributes["metadata"].items():
    #     metadata[k] = v
    if savetype == "ome":
        tifffile.imwrite(
            path,
            shape=layer_data.shape,
            dtype=layer_data.dtype,
            bigtiff=True,
            imagej=False,
            description=ome,
        )
        memmap = tifffile.memmap(path)
        for t in range(layer_data.shape[0]):
            for z in range(layer_data.shape[1]):
                for c in range(layer_data.shape[2]):
                    memmap[t, z, c, ...] = layer_data[t, z, c, ...]
                    memmap.flush()

    # # todo won't save shapes so that imageJ can open them correctly...
    #  https://forum.image.sc/t/writing-contiguous-ome-tiff-with-tifffile/
    elif savetype == "imagej":
        with tifffile.TiffWriter(path, bigtiff=True, imagej=False) as tif:
            tif.write(
                layer_data,
                resolution=(1 / px_size, 1 / px_size),
                # metadata=imagej_metadata,
                contiguous=True,
            )
    return [path]
