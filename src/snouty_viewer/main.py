import os
import sys
from glob import glob

from tifffile import tifffile

from src.snouty_viewer import im_container
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
    im_translated = xp.asarray(native_view(im))
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
    print()
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


def assign_memmory(dir_in, save_dir, format='native', save_name=None):
    file_list = glob(os.path.join(dir_in, 'data', '*.tif'))
    file_list.sort()
    num_files = len(file_list)
    frame_info = im_container.Im(dir_in, '000000')
    if format == 'raw':
        first_frame = frame_info.load_raw()
    elif format == 'native':
        first_frame = native_view(frame_info)
    elif format == 'traditional':
        first_frame = traditional_view(frame_info)
    else:
        print(f"[INFO] Format {format} is not a valid view type.")
        return
    dimension_order = 'TCZYX'
    shape = (num_files * first_frame.shape[0],) + first_frame.shape[1:]
    if save_name is None:
        save_name = 'merged'
    save_path = os.path.join(save_dir, save_name + f'-{format}.tif')
    tifffile.imwrite(save_path, shape=shape, dtype=first_frame.dtype, metadata={"axes": dimension_order})
    return save_path


def stitch_and_save(dir_in, save_dir, format='native'):
    save_path = assign_memmory(dir_in, save_dir, format=format)
    memmap = tifffile.memmap(save_path)
    file_list = glob(os.path.join(dir_in, 'data', '*.tif'))
    file_list.sort()
    memmap_start = 0
    for file in file_list:
        im_number = file.split('.tif')[-2][-6:]
        im_info = im_container.Im(dir_in, im_number)
        if format == 'raw':
            im = im_info.load_raw()
        elif format == 'native':
            im = native_view(im_info)
        elif format == 'traditional':
            im = traditional_view(im_info)
        else:
            print(f"[INFO] Format {format} is not a valid view type.")
            return
        memmap_end = memmap_start + im.shape[0]
        print(im_number)
        memmap[memmap_start:memmap_end, ...] = im
        memmap_start = memmap_end
    return save_path


if __name__ == "__main__":
    # root_dir = '/Users/austin/test_files/snouty_raw'
    # dir_list = os.listdir(root_dir)
    #
    # for sub_dir in dir_list[1:]:
    #     sub_dir_full = os.path.join(root_dir, sub_dir)
    #     save_dir = sub_dir_full
    #     stitch_and_save(sub_dir_full, save_dir)

    # TOP_DIR = "C:\\Users\\austin\\test_files\\2022-07-26_15-51-21_000_all_treatments_1hr_v2"
    TOP_DIR = "/Users/austin/test_files/snouty_raw/2022-04-21_16-52-33_000_mitotracker_ER-mEmerald/"
    # TOP_DIR = "/home/austin/Data/In/snouty_test"
    IM_NAME = "000000"
    #
    im_info = im_container.Im(TOP_DIR, IM_NAME)
    #
    # a = time.time()
    im_original = im_info.load_raw()
    # b = time.time()
    # print(f'[INFO] Raw image loaded in {b-a}s.')
    # im_preview = im_info.load_preview()
    # g = time.time()
    im_native = native_view(im_info)
    # print(f'[INFO] Preview image loaded in {g-b}s.')
    # d = time.time()
    # print(f'[INFO] Native view image loaded in {d-g}s.')
    im_traditional = traditional_view(im_info)
    # e = time.time()
    # print(f'[INFO] Traditional view image loaded in {e-d}s.')
    scale = (float(im_info.metadata['voxel_aspect_ratio']), 1, 1)
    # print('[INFO] Done')
    #
    #
    import napari
    viewer = napari.Viewer()
    viewer.add_image(im_original[:, 0, ...], scale=scale, colormap='viridis', name='original')
    viewer.add_image(im_native[:, 0, ...]+1, scale=scale, colormap='viridis', name='native')
    viewer.add_image(im_traditional[:, 0, ...]+1, colormap='viridis', name='traditional')
    # napari.run()
