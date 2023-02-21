from typing import Any, Dict, List

import tifffile


def assign_memory(layer, save_dir):
    layer_shape = layer.shape
    if len(layer_shape) == 5:
        dimension_order = "TCZYX"
    elif len(layer_shape) == 4:
        dimension_order = "TZYX"
    elif len(layer_shape) == 3:
        dimension_order = "ZYX"
    else:
        dimension_order = None
    tifffile.imwrite(
        save_dir,
        shape=layer_shape,
        dtype=layer.dtype,
        metadata={"axes": dimension_order},
    )
    return None


def write_single_image(
    path: str, layer_data: Any, attributes: Dict[str, Any]
) -> List[str]:
    assign_memory(layer_data, path)
    memmap = tifffile.memmap(path)
    memmap[:] = layer_data
    return [path]
