import sys

import napari
import numpy as np
import im_container
import skimage.transform


def interpolated_view(im: im_container.Im):
    im_raw = im.load_raw()
    num_t, num_c, num_z, num_y, num_x = im_raw.shape
    z_ratio = float(im.metadata['voxel_aspect_ratio'])
    scan_step_size_px = float(int(im.metadata['scan_step_size_px']))
    shear_angle = np.arctan(z_ratio)
    shear_length = int(np.rint(num_y * np.sin(shear_angle)))
    y_shrink = int(np.rint(shear_length / np.tan((np.pi-shear_angle)/2)))
    z_scaling = np.sqrt(scan_step_size_px**2+1)
    tform = skimage.transform.AffineTransform(shear=-shear_angle, scale=(z_scaling, 1))
    num_y_new = num_y - y_shrink
    num_z_new = int(np.rint((shear_length + num_z*z_scaling)))
    im_out = np.zeros((num_t, num_c, num_y_new, num_z_new, num_x), dtype=im_raw.dtype)
    swapped = np.swapaxes(im_raw, 2, 3)
    for t in range(num_t):
        sys.stdout.write(f"\r[INFO] Interpolating volume {t+1} of {num_t}...")
        sys.stdout.flush()
        for c in range(num_c):
            for x in range(num_x):
                im_out[t, c, :, :, x] = skimage.transform.warp(swapped[t, c, :, :, x], tform.inverse,
                                                               preserve_range=True,
                                                               output_shape=(num_y_new, num_z_new),
                                                               order=3)  # order 3 seems to be fastest
    return im_out


def translated_view(im: im_container.Im):
    im_raw = im.load_raw()
    num_t, num_c, num_z, num_y, num_x = im_raw.shape
    scan_step_size_px = int(im.metadata['scan_step_size_px'])
    max_deshear_shift = int(np.rint(scan_step_size_px * (num_z - 1)))
    im_desheared = np.zeros((num_t, num_c, num_z, num_y + max_deshear_shift, num_x), im_raw.dtype)
    for z in range(num_z):
        deshear_shift = int(np.rint(z * scan_step_size_px))
        im_desheared[:, :, z, deshear_shift:(deshear_shift + num_y), :] = im_raw[:, :, z, :, :]
    return im_desheared


if __name__ == "__main__":
    TOP_DIR = "/Users/austin/test_files/snouty_raw/2022-04-21_16-52-33_000_mitotracker_ER-mEmerald/"
    IM_NAME = "000000"

    im = im_container.Im(TOP_DIR, IM_NAME)

    im_raw = im.load_raw()
    im_preview = im.load_preview()
    im_translated = translated_view(im)
    im_interpolated = interpolated_view(im)

    z_ratio = float(im.metadata['voxel_aspect_ratio'])
    scale = (z_ratio, 1, 1)

    viewer = napari.Viewer()
    viewer.add_image(im_raw[:, 0, ...], scale=scale, contrast_limits=(0, 2000), colormap='viridis')
    viewer.add_image(im_translated[:, 0, ...], scale=scale, contrast_limits=(0, 2000), colormap='viridis')
    viewer.add_image(im_interpolated[:, 0, ...], contrast_limits=(0, 2000), colormap='viridis')
    napari.run()
