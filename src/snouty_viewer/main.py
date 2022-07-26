import sys
import im_container
import time
import numpy as np
try:
    import cupy as xp
    from cupyx.scipy import ndimage as ndi
    use_gpu = True
    print("[INFO] Using GPU.")
except ModuleNotFoundError:
    xp = np
    from scipy import ndimage as ndi
    use_gpu = False
    print("[INFO] Using CPU.")


def traditional_view(im: im_container.Im):  # a little smaller than native view, but much slower
    im_raw = im.load_raw()
    im_translated = native_view(im)
    num_t, num_c, num_z, num_y, num_x = im_raw.shape
    _, _, _, t_num_y, _ = im_translated.shape
    rotation_angle = float(im.metadata['tilt'])
    z_ratio = float(im.metadata['voxel_aspect_ratio'])

    scaled_z = num_z * z_ratio
    new_y = int(xp.rint(scaled_z*xp.sin(rotation_angle) + t_num_y*xp.cos(rotation_angle)))
    new_z = int(xp.rint(num_y*xp.sin(rotation_angle)))
    temp_z = int(xp.rint(scaled_z*xp.cos(rotation_angle) + t_num_y*xp.sin(rotation_angle)))
    start_z = (temp_z-new_z)//2

    im_out = np.zeros((num_t, num_c, new_z, new_y, num_x), dtype=im_raw.dtype)
    for t in range(num_t):
        sys.stdout.write(f"\r[INFO] Interpolating volume {t+1} of {num_t}...")
        sys.stdout.flush()
        for c in range(num_c):
            temp1 = ndi.zoom(im_translated[t, c, ...], zoom=(z_ratio, 1, 1),
                                       order=1, prefilter=True)
            temp2 = ndi.rotate(temp1, xp.rad2deg(rotation_angle),
                                         order=1, prefilter=True, reshape=True)
            if use_gpu:
                im_out[t, c, ...] = temp2[start_z:(start_z+new_z), ...].get()
            else:
                im_out[t, c, ...] = temp2[start_z:(start_z + new_z), ...]
    return im_out


def native_view(im: im_container.Im):  # a little bigger than traditional view, but much faster
    im_raw = im.load_raw()
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


if __name__ == "__main__":
    TOP_DIR = "/Users/austin/test_files/snouty_raw/2022-04-21_16-52-33_000_mitotracker_ER-mEmerald/"
    # TOP_DIR = "/home/austin/Data/In/snouty_test"
    IM_NAME = "000000"

    im_info = im_container.Im(TOP_DIR, IM_NAME)

    a = time.time()
    im_original = im_info.load_raw()
    b = time.time()
    print(f'[INFO] Raw image loaded in {b-a}s.')
    im_preview = im_info.load_preview()
    g = time.time()
    im_native = native_view(im_info)
    print(f'[INFO] Preview image loaded in {g-b}s.')
    d = time.time()
    print(f'[INFO] Native view image loaded in {d-g}s.')
    im_traditional = traditional_view(im_info)
    e = time.time()
    print(f'Native view image loaded in {e-d}s.')
    scale = (float(im_info.metadata['voxel_aspect_ratio']), 1, 1)
    print('[INFO] Done')


    # import napari
    # viewer = napari.Viewer()
    # viewer.add_image(im_original[:, 0, ...]+1, scale=scale, colormap='viridis', name='original')
    # viewer.add_image(im_native[:, 0, ...]+1, scale=scale, colormap='viridis', name='native')
    # viewer.add_image(im_traditional[:, 0, ...]+1, colormap='viridis', name='traditional')
    # napari.run()
