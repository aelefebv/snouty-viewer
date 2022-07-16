import os

import napari
import numpy as np
import tifffile
import scipy.ndimage
import im_container
from time import time
import skimage.transform


# def desheared_rotate(im: im_container.Im):
#     im_raw = im.load_raw()
#     num_t, num_c, num_z, num_y, num_x = im_raw.shape
#     scan_step_size_px = int(im.metadata['scan_step_size_px'])
#     max_deshear_shift = int(np.rint(scan_step_size_px * (num_z - 1)))
#     im_desheared = np.zeros((num_t, num_c, num_z, num_y + max_deshear_shift, num_x), im_raw.dtype)
#     shear_angle = np.arctan(1/scan_step_size_px)
#     tform = skimage.transform.AffineTransform(shear=-shear_angle)
#
#
#     for z in range(num_z):
#         deshear_shift = int(np.rint(z * scan_step_size_px))
#         im_desheared[:, :, z, deshear_shift:(deshear_shift + num_y), :] = im_raw[:, :, z, :, :]
#         im_out[t, c, :, :, x] = skimage.transform.warp(swapped[t, c, :, :, x], tform.inverse,
#                                                        preserve_range=True, output_shape=(num_y_new, num_z_new))
#     return im_desheared


def affine(im: im_container.Im):
    im_raw = im.load_raw()
    num_t, num_c, num_z, num_y, num_x = im_raw.shape
    z_ratio = float(im.metadata['voxel_aspect_ratio'])
    scan_step_size_px = float(int(im.metadata['scan_step_size_px']))
    # scan_step_size_px = 2
    shear_angle = np.arctan(z_ratio)
    # shear_angle = np.deg2rad(30)
    shear_length = int(np.rint(num_y * np.sin(shear_angle)))
    y_shrink = int(np.rint(shear_length / np.tan((np.pi-shear_angle)/2)))
    z_scaling = np.sqrt(scan_step_size_px**2+1)
    tform = skimage.transform.AffineTransform(shear=-shear_angle, scale=(z_scaling, 1))
    num_y_new = num_y - y_shrink
    num_z_new = int(np.rint((shear_length + num_z*z_scaling)))
    im_out = np.zeros((num_t, num_c, num_y_new, num_z_new, num_x), dtype=im_raw.dtype)
    swapped = np.swapaxes(im_raw, 2, 3)
    for t in range(num_t):
        print(t, num_t)
        for c in range(num_c):
            for x in range(num_x):
                im_out[t, c, :, :, x] = skimage.transform.warp(swapped[t, c, :, :, x], tform.inverse,
                                                               preserve_range=True,
                                                               output_shape=(num_y_new, num_z_new),
                                                               order=3)
    return im_out


# def z_deshear(im: im_container.Im):
#     im_raw = im.load_raw()
#     num_t, num_c, num_z, num_y, num_x = im_raw.shape
#     scan_step_size_px = int(im.metadata['scan_step_size_px'])
#     z_shift = scan_step_size_px * np.cos(np.deg2rad(30))
#     max_z_shift = int(np.rint(z_shift * (num_y-1)))
#     im_desheared = np.zeros((num_t, num_c, num_z + max_z_shift, num_y, num_x), im_raw.dtype)
#     im_desheared[:, :, :num_z, :, :] = im_raw
#     for y in range(140, 150): # range(num_y):
#         print(y)
#         deshear_shift = y * z_shift
#         integer_shift = int(np.rint(deshear_shift))
#         print(deshear_shift, integer_shift)
#         print(integer_shift, (integer_shift + num_z))
#         print((integer_shift + num_z)-integer_shift, im_raw[..., y, :].shape[2])
#         # , output = im_desheared[:, :, integer_shift:(integer_shift + num_z), y, :]
#         test = scipy.ndimage.shift(im_desheared[:, :, 0:num_z, y, :], (0, 0, z_shift, 0))
#         im_desheared[:, :, integer_shift:(integer_shift + num_z), y, :] = test.copy()
#     return im_desheared


def z_deshear(im):
    im_raw = im.load_raw()
    num_t, num_c, num_z, num_y, num_x = im_raw.shape
    z_ratio = float(im.metadata['voxel_aspect_ratio'])
    scan_step_size_px = int(im.metadata['scan_step_size_px'])
    max_deshear_shift = int(np.rint((num_y-1)/scan_step_size_px))
    im_desheared = np.zeros((num_t, num_c, (num_z + max_deshear_shift), num_y, num_x), im_raw.dtype)
    for idx, y in enumerate(range(0, num_y, scan_step_size_px)):
        deshear_shift = idx
        print(deshear_shift, y)
        im_desheared[:, :, deshear_shift:(deshear_shift+num_z), y:(y+scan_step_size_px), :] = im_raw[:, :, :, y:(y+scan_step_size_px), :]
    return im_desheared


def z_deshear_bk(im):
    im_raw = im.load_raw()
    num_t, num_c, num_z, num_y, num_x = im_raw.shape
    scan_step_size_px = int(im.metadata['scan_step_size_px'])
    max_deshear_shift = int(np.rint((num_y - 1) / scan_step_size_px))
    im_desheared = np.zeros((num_t, num_c, num_z + max_deshear_shift, num_y, num_x), im_raw.dtype)
    for y in range(0, num_y, scan_step_size_px):
        deshear_shift = int(np.rint(y / scan_step_size_px))
        im_desheared[:, :, deshear_shift:(deshear_shift + num_z), y:(y + scan_step_size_px), :] = im_raw[:, :, :, y:(
                    y + scan_step_size_px), :]
    return im_desheared


def desheared(im: im_container.Im):
    im_raw = im.load_raw()
    num_t, num_c, num_z, num_y, num_x = im_raw.shape
    scan_step_size_px = int(im.metadata['scan_step_size_px'])
    max_deshear_shift = int(np.rint(scan_step_size_px * (num_z - 1)))
    im_desheared = np.zeros((num_t, num_c, num_z, num_y + max_deshear_shift, num_x), im_raw.dtype)
    for z in range(num_z):
        deshear_shift = int(np.rint(z * scan_step_size_px))
        im_desheared[:, :, z, deshear_shift:(deshear_shift + num_y), :] = im_raw[:, :, z, :, :]
    return im_desheared



def deshear(im_container, im):
    num_t, num_c, num_z, num_y, num_x = im.shape
    scan_step_size_px = int(im_container.metadata['scan_step_size_px'])
    rotation_rad = np.arctan(1/scan_step_size_px)
    max_deshear_shift_y = int(np.rint(scan_step_size_px * (num_y - 1) * np.cos(rotation_rad)))
    max_deshear_shift_z = int(np.rint(scan_step_size_px * (num_z - 1) * np.sin(rotation_rad)))
    im_desheared = np.zeros((num_t, num_c, num_z + max_deshear_shift_z, num_y + max_deshear_shift_y, num_x), im.dtype)
    # shift z by sine of angle, y by cosine of angle
    for z in range(num_z):
        for y in range(num_y):
            deshear_shift_y = int(np.rint(y * scan_step_size_px*np.cos(rotation_rad)))
            deshear_shift_z = int(np.rint(z * scan_step_size_px*np.sin(rotation_rad)))
            im_desheared[:, :, z - deshear_shift_z, y - deshear_shift_y, :] = im[:, :, z, y, :]
    return im_desheared


def traditional(im: im_container.Im):
    im_desheared = desheared(im)
    og_num_y = int(im.metadata['height_px'])
    im_desheared = im_desheared[:, :, :, (og_num_y//2):(-og_num_y//2)]
    num_t, num_c, num_z, num_y, num_x = im_desheared.shape
    scan_step_size_px = int(im.metadata['scan_step_size_px'])
    rotation_rad = np.arctan(1/scan_step_size_px)
    new_num_y = int(np.rint(num_z*np.sin(rotation_rad) + num_y*np.cos(rotation_rad)))
    new_num_z = int(np.rint(num_z*np.cos(rotation_rad) + num_y*np.sin(rotation_rad)))
    im_native = np.zeros((num_t, num_c, new_num_z, new_num_y, num_x), dtype=im_desheared.dtype)
    im_native = np.zeros((1, 1, new_num_z, new_num_y, num_x), dtype=im_desheared.dtype)
    scipy.ndimage.rotate(im_desheared[:1, :1, ...], angle=np.rad2deg(rotation_rad), output=im_native, axes=(2, 3))
    # for t in range(num_t):
    #     print(f"rotating volume {t} of {num_t}")
    #     for c in range(num_c):
    #         scipy.ndimage.rotate(im_desheared[t, c, ...], angle=np.rad2deg(rotation_rad), output = im_native[t, c, ...])
    final_z = int(np.rint(og_num_y * np.sin(rotation_rad)))
    crop = new_num_z - final_z
    im_native = im_native[:, :, (crop//2):(-crop//2), :, :]
    return im_native


def rotate_deshear(im: im_container.Im):
    im_raw = im.load_raw()
    og_num_y = int(im.metadata['height_px'])
    num_t, num_c, num_z, num_y, num_x = im_raw.shape
    scan_step_size_px = int(im.metadata['scan_step_size_px'])
    rotation_rad = np.arctan(1 / scan_step_size_px)
    new_num_y = int(np.rint(num_z * np.sin(rotation_rad) + num_y * np.cos(rotation_rad)))
    new_num_z = int(np.rint(num_z * np.cos(rotation_rad) + num_y * np.sin(rotation_rad)))
    im_native = np.zeros((num_t, num_c, new_num_z, new_num_y, num_x), dtype=im_raw.dtype)
    im_native = np.zeros((2, 2, new_num_z, new_num_y, num_x), dtype=im_raw.dtype)
    scipy.ndimage.rotate(im_raw[:2, :2, ...], angle=np.rad2deg(rotation_rad), output=im_native, axes=(2, 3))
    # for t in range(num_t):
    #     print(f"rotating volume {t} of {num_t}")
    #     for c in range(num_c):
    #         scipy.ndimage.rotate(im_desheared[t, c, ...], angle=np.rad2deg(rotation_rad), output = im_native[t, c, ...])
    final_z = int(np.rint(og_num_y * np.sin(rotation_rad)))
    crop = new_num_z - final_z
    # im_native = im_native[:, :, (crop // 2):(-crop // 2), :, :]
    return im_native


if __name__ == "__main__":
    TOP_DIR = "/Users/austin/test_files/snouty_raw/2022-04-21_16-52-33_000_mitotracker_ER-mEmerald/"
    IM_NAME = "000000"

    im = im_container.Im(TOP_DIR, IM_NAME)

    im_raw = im.load_raw()
    im_preview = im.load_preview()
    im_desheared_y = desheared(im)
    # im_desheared_z = z_deshear(im)
    affine_test = affine(im) + 1
    # im_native = traditional(im)
    # im_traditional_1 = traditional(im)
    # test_rotate = scipy.ndimage.rotate(im_desheared[0, 0, ...], reshape=True, angle=np.rad2deg(np.arctan(1/3)), axes=(0, 1)) + 1

    # im_native_desheared = rotate_deshear(im)
    # im_native_desheared_actually = deshear(im, im_native_desheared)


    z_ratio = float(im.metadata['voxel_aspect_ratio'])
    scale = (z_ratio, 1, 1)
    viewer = napari.Viewer()
    viewer.add_image(im_raw[:, 0, ...]+1, scale=scale, contrast_limits=(0, 2000), colormap='viridis')
    # viewer.add_image(im_raw[:, 1, ...], contrast_limits=(0, 2000), scale=scale, colormap='viridis')
    # viewer.add_image(im_preview[:, 0, ...])
    # viewer.add_image(im_preview[:, 1, ...])
    viewer.add_image(im_desheared_y[:, 0, ...]+1, scale=scale, contrast_limits=(0, 2000), colormap='viridis')
    # viewer.add_image(im_desheared_z[:, 0, ...]+1, scale=(1, 3.16, 1, 1), contrast_limits=(0, 2000), colormap='viridis')
    # viewer.add_image(np.swapaxes(im_raw[0, 0, :, :, 200], 0, 1)+1, contrast_limits=(0, 2000), colormap='viridis')
    # viewer.add_image(affine_test, contrast_limits=(0, 2000), colormap='viridis')
    viewer.add_image(affine_test[:, 0, ...], contrast_limits=(0, 2000), colormap='viridis')
    # viewer.add_image(im_desheared[:, 1, ...], contrast_limits=(0, 2000), colormap='viridis')
    # viewer.add_image(im_desheared[:, 1, ...], contrast_limits=(0, 2000), scale=scale, colormap='viridis')
    # viewer.add_image(im_native+1, contrast_limits=(0, 2000), colormap='viridis')
    # viewer.add_image(im_native_desheared+1, contrast_limits=(0, 2000), colormap='viridis')
    # viewer.add_image(im_native_desheared_actually+1, contrast_limits=(0, 2000), colormap='viridis')
    # viewer.add_image(im_traditional[:, 1, ...], contrast_limits=(0, 2000), scale=scale, colormap='viridis')
    napari.run()
