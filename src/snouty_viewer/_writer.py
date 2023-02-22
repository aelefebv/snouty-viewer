from typing import Any, Dict, List

import numpy as np
import tifffile


def write_single_image(
    path: str, layer_data: Any, attributes: Dict
) -> List[str]:
    if len(layer_data.shape) == 4:
        layer_data = np.expand_dims(layer_data, axis=1)
    px_size = float(attributes["metadata"]["snouty_metadata"]["sample_px_um"])
    z_px_size = px_size * float(
        attributes["metadata"]["snouty_metadata"]["voxel_aspect_ratio"]
    )
    vps = float(attributes["metadata"]["snouty_metadata"]["volumes_per_s"])
    spb = float(attributes["metadata"]["snouty_metadata"]["buffer_time_s"])
    metadata = {
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
        "fps": vps + 1 / spb,
    }
    for k, v in attributes["metadata"].items():
        metadata[k] = v
    print(metadata)
    with tifffile.TiffWriter(path, bigtiff=True, imagej=True) as tif:
        tif.write(
            np.swapaxes(layer_data, 1, 2),
            resolution=(1 / px_size, 1 / px_size),
            metadata=metadata,
        )
    return [path]
