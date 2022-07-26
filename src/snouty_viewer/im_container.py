import os

from tifffile import tifffile
try:
    import cupy as np
    from cupyx.scipy import ndimage as ndi
except ModuleNotFoundError:
    import numpy as np
    from scipy import ndimage as ndi


class Im:
    def __init__(self, top_dir, im_name):
        self.raw_path = os.path.join(top_dir, 'data', im_name + '.tif')
        self.preview_path = os.path.join(top_dir, 'preview', im_name + '.tif')
        self.metadata_path = os.path.join(top_dir, 'metadata', im_name + '.txt')
        self.metadata = self.load_metadata()

    def load_metadata(self):
        with open(self.metadata_path) as f:
            lines = f.readlines()
        metadata_dict = dict()
        for line in lines:
            split = line.split(': ')
            metadata_dict[split[0]] = split[1][:-1]
        return metadata_dict

    def load_raw(self, remove_timebar=True):
        loaded_im = tifffile.imread(self.raw_path)  # reads in as TZCYX
        tczyx = np.swapaxes(loaded_im, 1, 2)  # flips to TCZYX for Napari
        if remove_timebar:
            tczyx = tczyx[..., 8:, :]
        return tczyx

    def load_preview(self):
        loaded_im = tifffile.imread(self.preview_path)
        return loaded_im
