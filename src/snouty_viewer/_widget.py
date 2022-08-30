"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget

if TYPE_CHECKING:
    import napari


class ExampleQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btn)

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")


@magic_factory
def native_view(im: im_container.Im):  # a little bigger than traditional view, but much faster
    num_t, num_c, num_z, num_y, num_x = im_raw.shape
    scan_step_size_px = int(im.metadata['scan_step_size_px'])
    max_deshear_shift = int(xp.rint(scan_step_size_px * (num_z - 1)))
    im_desheared = xp.zeros((num_t, num_c, num_z, num_y + max_deshear_shift, num_x), im_raw.dtype)
    for z in range(num_z):
        deshear_shift = int(xp.rint(z * scan_step_size_px))
        im_desheared[:, :, z, deshear_shift:(deshear_shift + num_y), :] = im_raw[:, :, z, :, :]
    if use_gpu:
        return im_desheared.get()
    else:
        return im_desheared


# Uses the `autogenerate: true` flag in the plugin manifest
# to indicate it should be wrapped as a magicgui to autogenerate
# a widget.
def example_function_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")
