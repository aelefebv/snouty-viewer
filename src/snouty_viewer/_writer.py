from typing import Any, Dict, List

import numpy as np
import ome_types
import tifffile


def write_single_image(
    path: str, layer_data: Any, attributes: Dict
) -> List[str]:
    if len(layer_data.shape) == 4:
        layer_data = np.expand_dims(layer_data, axis=1)
    layer_data = np.swapaxes(layer_data, 1, 2)
    tifffile.imwrite(
        path, layer_data, bigtiff=True, metadata={"axes": "TZCYX"}
    )
    ome_xml = tifffile.tiffcomment(path)
    ome = ome_types.from_xml(ome_xml, parser="lxml")

    snouty_metadata = attributes["metadata"]["snouty_metadata"]
    px_size = float(snouty_metadata["sample_px_um"])
    z_px_size = px_size * float(snouty_metadata["voxel_aspect_ratio"])
    ome.images[0].pixels.physical_size_x = px_size
    ome.images[0].pixels.physical_size_y = px_size
    ome.images[0].pixels.physical_size_z = z_px_size

    vps = float(snouty_metadata["volumes_per_s"])
    spb = float(snouty_metadata["buffer_time_s"])
    delay = snouty_metadata["delay_s"]
    if delay is None or delay == "None":
        delay = 0.0
    else:
        delay = float(delay)
    time_increment = 1 / vps + spb + delay
    ome.images[0].pixels.time_increment = time_increment

    # todo fix datetime format to work
    # acquisition_date = f"{snouty_metadata['Date']} - " \
    #                    f"{snouty_metadata['Time']}"
    # ome.images[0].acquisition_date = acquisition_date
    description = snouty_metadata["description"]
    ome.images[0].description = description
    ome_xml = ome.to_xml()
    tifffile.tiffcomment(path, ome_xml)
    return [path]
