import numpy as np

from snouty_viewer import native_view


def test_3d_image(make_napari_viewer):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.random((20, 20, 20)))
    layer.metadata["snouty_metadata"] = {"scan_step_size_px": 3}
    deshear_size = 3 * 19
    out_size = (1, 20, 20 + deshear_size, 20)

    # this time, our widget will be a MagicFactory or FunctionGui instance
    my_widget = native_view()

    # if we "call" this object, it'll execute our function
    output_layer = my_widget(viewer.layers[0])

    # read captured output and check that it's as we expected
    assert output_layer[0].shape == out_size


def test_4d_image(make_napari_viewer):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.random((4, 20, 20, 20)))
    layer.metadata["snouty_metadata"] = {"scan_step_size_px": 3}
    deshear_size = 3 * 19
    out_size = (4, 20, 20 + deshear_size, 20)

    # this time, our widget will be a MagicFactory or FunctionGui instance
    my_widget = native_view()

    # if we "call" this object, it'll execute our function
    output_layer = my_widget(viewer.layers[0])

    # read captured output and check that it's as we expected
    assert output_layer[0].shape == out_size
