"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/plugins/guides.html?#writers

Replace code below according to your needs.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, List, Sequence, Tuple, Union
import tifffile
from napari.qt import thread_worker


if TYPE_CHECKING:
    import napari
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = Tuple[DataType, dict, str]


def write_single_image(path: str, layer: Any, meta: dict):
    save_path = os.path.join(path, layer.name) + '.tif'
    @thread_worker()
    def save_im(im_layer, metadata):
        tifffile.imwrite(save_path, im_layer.data, metadata=metadata)

    worker = save_im(layer, save_path, meta)
    worker.start()
    return save_path


def write_multiple(path: str, data: List[FullLayerData]):
    """Writes multiple layers of different types."""
