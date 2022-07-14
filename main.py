import os

import napari
import numpy as np
import tifffile
import scipy.ndimage


# todo copied over from alfred's. understand it and modify as needed:
# ----
# SOLS optical configuration (edit as needed):
M1 = 200 / 2
Mscan = 70 / 70
M2 = 5 / 357
M3 = 200 / 5
MRR = M1 * Mscan * M2
Mtot = MRR * M3
camera_px_um = 6.5
sample_px_um = camera_px_um / Mtot
tilt = np.deg2rad(30)


# native view
# The 'native view' is the most principled view of the data for analysis.
# If 'type(scan_step_size_px) is int' (default) then no interpolation is
# needed to view the volume. The native view looks at the sample with
# the 'tilt' of the Snouty objective (microsope 3 in the emmission path).
def get(
    self,
    data, # raw 5D data, 'tzcyx' input -> 'tzcyx' output
    scan_step_size_px):
    vo, slices, ch, h_px, w_px = data.shape
    prop_px = h_px # light-sheet propagation axis
    scan_step_px_max = int(np.rint(scan_step_size_px * (slices - 1)))
    data_native = np.zeros(
        (vo, slices, ch, prop_px + scan_step_px_max, w_px), 'uint16')
    for v in range(vo):
        for c in range(ch):
            for i in range(slices):
                prop_px_shear = int(np.rint(i * scan_step_size_px))
                data_native[
                    v, i, c, prop_px_shear:prop_px + prop_px_shear, :] = (
                        data[v, i, c, :, :])
    return data_native # larger!


def calculate_voxel_aspect_ratio(scan_step_size_px):
    return scan_step_size_px * np.tan(tilt)


# traditional view
# Very slow but pleasing - rotates the native view to the traditional view!
def get(
    self,
    data_native, # raw 5D data, 'tzcyx' input -> 'tzcyx' output
    scan_step_size_px):
    vo, slices, ch, h_px, w_px = data_native.shape
    voxel_aspect_ratio = calculate_voxel_aspect_ratio(scan_step_size_px)
    tzcyx = []
    for v in range(vo):
        zcyx = []
        for c in range(ch):
            zyx_native_cubic_voxels = scipy.ndimage.zoom(
                data_native[v, :, c, :, :], (voxel_aspect_ratio, 1, 1))
            zyx_traditional = scipy.ndimage.rotate(
                zyx_native_cubic_voxels, np.rad2deg(tilt))
            zcyx.append(zyx_traditional[:, np.newaxis, : ,:])
        zcyx = np.concatenate(zcyx, axis=1)
        tzcyx.append(zcyx[np.newaxis, :, :, : ,:])
    data_traditional = np.concatenate(tzcyx, axis=0)
    return data_traditional # even larger!
# ------


def load_raw(im_path, remove_timebar=True):
    loaded_im = tifffile.imread(im_path)  # reads in as TZCYX
    tczyx = np.swapaxes(loaded_im, 1, 2)  # flips to TCZYX
    if remove_timebar:
        tczyx = tczyx[..., 8:, :]
    return tczyx


def load_preview(im_path):
    loaded_im = tifffile.imread(im_path)
    return loaded_im


if __name__ == "__main__":
    top_dir = "/Users/austin/test_files/snouty_raw/"
    im_name = "000000.tif"

    im_raw = load_raw(os.path.join(top_dir, 'data', im_name))
    im_preview = load_preview(os.path.join(top_dir, 'preview', im_name))
    viewer = napari.Viewer()
    viewer.add_image(im_raw[:, 0, ...])
    viewer.add_image(im_raw[:, 1, ...])
    viewer.add_image(im_preview[:, 0, ...])
    viewer.add_image(im_preview[:, 1, ...])
    napari.run()
