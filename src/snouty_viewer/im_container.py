import os

from tifffile import tifffile
import numpy as np
try:
    import cupy as xp
    from cupyx.scipy import ndimage as ndi
except ModuleNotFoundError:
    xp = np
    from scipy import ndimage as ndi


class Im:
    def __init__(self, top_dir, im_name=None):
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
        swap = True
        if len(loaded_im.shape == 4):
            loaded_im = np.swapaxes(loaded_im, 0, 1)
        while len(loaded_im.shape) < 5:
            swap = ~swap
            loaded_im = np.expand_dims(loaded_im, axis=0)
        if swap is True:
            tczyx = np.swapaxes(loaded_im, 1, 2)  # flips to TCZYX for Napari
        else:
            tczyx = loaded_im
        if remove_timebar:
            tczyx = tczyx[..., 8:, :]
        return xp.asarray(tczyx)

    def load_preview(self):
        loaded_im = tifffile.imread(self.preview_path)
        return xp.asarray(loaded_im)
