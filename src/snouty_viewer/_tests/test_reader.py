import numpy as np
import os
import tifffile

from snouty_viewer import napari_get_reader


# tmp_path is a pytest fixture
def test_reader_single_channel(tmp_path):
    """An example of how you might test your plugin."""

    # write some fake data using your supported file format
    metadata_directory = tmp_path / "metadata"
    metadata_directory.mkdir()
    data_directory = tmp_path / "data"
    data_directory.mkdir()
    my_test_file = str(data_directory / "000000.tif")
    my_test_meta = metadata_directory / "000000.txt"

    volumes_per_buffer = 1
    num_buffers = 1
    slices_per_volume = 5

    needed_metadata = [f'volumes_per_buffer: {volumes_per_buffer}', 'channels_per_slice: [\'488\']', f'slices_per_volume: {slices_per_volume}']

    # os.mkdir(metadata_directory)
    with open(my_test_meta, 'w+') as f:
        for line in needed_metadata:
            f.write(line)
            f.write('\n')

    original_data = np.random.rand(slices_per_volume, 20, 20)
    tifffile.imwrite(my_test_file, original_data)

    # try to read it back in
    reader = napari_get_reader(tmp_path)
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data_list = reader(str(tmp_path))
    assert isinstance(layer_data_list, list)

    # make sure it's the same as it started
    modified_shape = (num_buffers, original_data.shape[0], original_data.shape[1]-8, original_data.shape[2])
    assert modified_shape == layer_data_list[0][0].shape

def test_reader_multi_channel(tmp_path):
    """An example of how you might test your plugin."""

    # write some fake data using your supported file format
    metadata_directory = tmp_path / "metadata"
    metadata_directory.mkdir()
    data_directory = tmp_path / "data"
    data_directory.mkdir()
    my_test_file = str(data_directory / "000000.tif")
    my_test_meta = metadata_directory / "000000.txt"

    volumes_per_buffer = 1
    num_buffers = 1
    slices_per_volume = 5

    needed_metadata = [f'volumes_per_buffer: {volumes_per_buffer}', 'channels_per_slice: [\'488\', \'405\']', f'slices_per_volume: {slices_per_volume}']

    # os.mkdir(metadata_directory)
    with open(my_test_meta, 'w+') as f:
        for line in needed_metadata:
            f.write(line)
            f.write('\n')

    original_data = np.random.rand(volumes_per_buffer, slices_per_volume, 2, 20, 20)
    tifffile.imwrite(my_test_file, original_data)

    # try to read it back in
    reader = napari_get_reader(tmp_path)
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data_list = reader(str(tmp_path))
    assert isinstance(layer_data_list, list)

    # make sure it's the same as it started
    modified_shape = (num_buffers, original_data.shape[1], original_data.shape[3]-8, original_data.shape[4])
    assert modified_shape == layer_data_list[0][0].shape
    assert modified_shape == layer_data_list[1][0].shape

def test_reader_multi_channel_multi_file(tmp_path):
    """An example of how you might test your plugin."""

    # write some fake data using your supported file format
    metadata_directory = tmp_path / "metadata"
    metadata_directory.mkdir()
    data_directory = tmp_path / "data"
    data_directory.mkdir()
    my_test_file0 = str(data_directory / "000000.tif")
    my_test_file1 = str(data_directory / "000001.tif")
    my_test_meta = metadata_directory / "000000.txt"

    volumes_per_buffer = 1
    num_buffers = 2
    slices_per_volume = 5

    needed_metadata = [f'volumes_per_buffer: {volumes_per_buffer}', 'channels_per_slice: [\'488\', \'405\']', f'slices_per_volume: {slices_per_volume}']

    # os.mkdir(metadata_directory)
    with open(my_test_meta, 'w+') as f:
        for line in needed_metadata:
            f.write(line)
            f.write('\n')

    original_data0 = np.random.rand(volumes_per_buffer, slices_per_volume, 2, 20, 20)
    original_data1 = np.random.rand(volumes_per_buffer, slices_per_volume, 2, 20, 20)
    tifffile.imwrite(my_test_file0, original_data0)
    tifffile.imwrite(my_test_file1, original_data1)

    # try to read it back in
    reader = napari_get_reader(tmp_path)
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data_list = reader(str(tmp_path))
    assert isinstance(layer_data_list, list)

    # make sure it's the same as it started
    modified_shape = (num_buffers, original_data0.shape[1], original_data0.shape[3]-8, original_data0.shape[4])
    assert modified_shape == layer_data_list[0][0].shape
    assert modified_shape == layer_data_list[1][0].shape

def test_reader_multi_channel_multi_vol(tmp_path):
    """An example of how you might test your plugin."""

    # write some fake data using your supported file format
    metadata_directory = tmp_path / "metadata"
    metadata_directory.mkdir()
    data_directory = tmp_path / "data"
    data_directory.mkdir()
    my_test_file0 = str(data_directory / "000000.tif")
    my_test_file1 = str(data_directory / "000001.tif")
    my_test_meta = metadata_directory / "000000.txt"

    volumes_per_buffer = 2
    num_buffers = 2
    slices_per_volume = 5

    needed_metadata = [f'volumes_per_buffer: {volumes_per_buffer}', 'channels_per_slice: [\'488\', \'405\']',
                       f'slices_per_volume: {slices_per_volume}']

    # os.mkdir(metadata_directory)
    with open(my_test_meta, 'w+') as f:
        for line in needed_metadata:
            f.write(line)
            f.write('\n')

    original_data0 = np.random.rand(volumes_per_buffer, slices_per_volume, 2, 20, 20)
    original_data1 = np.random.rand(volumes_per_buffer, slices_per_volume, 2, 20, 20)
    tifffile.imwrite(my_test_file0, original_data0)
    tifffile.imwrite(my_test_file1, original_data1)

    # try to read it back in
    reader = napari_get_reader(tmp_path)
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data_list = reader(str(tmp_path))
    assert isinstance(layer_data_list, list)

    # make sure it's the same as it started
    modified_shape = (num_buffers*volumes_per_buffer, original_data0.shape[1], original_data0.shape[3] - 8, original_data0.shape[4])
    assert modified_shape == layer_data_list[0][0].shape
    assert modified_shape == layer_data_list[1][0].shape

def test_reader_single_channel_multi_vol(tmp_path):
    """An example of how you might test your plugin."""

    # write some fake data using your supported file format
    metadata_directory = tmp_path / "metadata"
    metadata_directory.mkdir()
    data_directory = tmp_path / "data"
    data_directory.mkdir()
    my_test_file0 = str(data_directory / "000000.tif")
    my_test_file1 = str(data_directory / "000001.tif")
    my_test_meta = metadata_directory / "000000.txt"

    volumes_per_buffer = 2
    num_buffers = 2
    slices_per_volume = 5

    needed_metadata = [f'volumes_per_buffer: {volumes_per_buffer}', 'channels_per_slice: [\'488\']',
                       f'slices_per_volume: {slices_per_volume}']

    # os.mkdir(metadata_directory)
    with open(my_test_meta, 'w+') as f:
        for line in needed_metadata:
            f.write(line)
            f.write('\n')

    original_data0 = np.random.rand(volumes_per_buffer, slices_per_volume, 20, 20)
    original_data1 = np.random.rand(volumes_per_buffer, slices_per_volume, 20, 20)
    tifffile.imwrite(my_test_file0, original_data0)
    tifffile.imwrite(my_test_file1, original_data1)

    # try to read it back in
    reader = napari_get_reader(tmp_path)
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data_list = reader(str(tmp_path))
    assert isinstance(layer_data_list, list)

    # make sure it's the same as it started
    modified_shape = (num_buffers*volumes_per_buffer, original_data0.shape[1], original_data0.shape[2]-8, original_data0.shape[3])
    assert modified_shape == layer_data_list[0][0].shape


def test_get_reader_pass():
    reader = napari_get_reader("fake.file")
    assert reader is None
