import os
import numpy as np
import medpy.io as medio  # A module from the MedPy library for medical image processing
from multiprocessing import Pool
from itertools import repeat


def ensure_minimum_dimension(xmin, xmax, min_size=128):
    """Ensure the dimension size is at least 'min_size'."""
    if xmax - xmin < min_size:
        print('#' * 100)
        padding = int((min_size - (xmax - xmin)) / 2)
        xmax = xmax + padding + 1
        xmin = xmin - padding
    if xmin < 0:
        xmax -= xmin
        xmin = 0
    return xmin, xmax


def crop_volume(volume):
    """
    Crop the volume to remove the background and ensure the dimensions are at least 128.
    Args:
        volume: 3D numpy array representing the volume data.

    Returns:
        Cropped indices for each axis: x_min, x_max, y_min, y_max, z_min, z_max
    """
    if len(volume.shape) == 4:
        volume = np.amax(volume, axis=0)

    assert len(volume.shape) == 3, "Expected a 3D volume"

    x_dim, y_dim, z_dim = volume.shape
    non_zero_coords = np.where(volume != 0)

    x_min, x_max = np.amin(non_zero_coords[0]), np.amax(non_zero_coords[0])
    y_min, y_max = np.amin(non_zero_coords[1]), np.amax(non_zero_coords[1])
    z_min, z_max = np.amin(non_zero_coords[2]), np.amax(non_zero_coords[2])

    x_min, x_max = ensure_minimum_dimension(x_min, x_max)
    y_min, y_max = ensure_minimum_dimension(y_min, y_max)
    z_min, z_max = ensure_minimum_dimension(z_min, z_max)

    return x_min, x_max, y_min, y_max, z_min, z_max


def normalize_volume(volume):
    """
    Normalize the volume by zero-mean and unit-variance using non-zero elements as reference.

    Args:
        volume: 3D numpy array representing the volume data.

    Returns:
        Normalized volume
    """
    mask = volume.sum(axis=0) > 0  # Identify non-zero elements
    for k in range(3):  # Iterate over the channels (T1c, T1, T2)
        channel = volume[k, ...]
        non_zero_values = channel[mask]
        channel = (channel - non_zero_values.mean()) / non_zero_values.std()
        volume[k, ...] = channel

    return volume


def preprocess_volume(file_name, src_path, tar_path):
    """
    Preprocess the volume data by loading, cropping, normalizing, and saving it.

    Args:
        file_name: The name of the file (without extension) to preprocess.
        src_path: The source directory where the original files are located.
        tar_path: The target directory to save the processed file.
    """
    # Skip if preprocessed file already exists
    if os.path.exists(os.path.join(tar_path, f'{file_name}_vol.npy')):
        return

    # Load the MRI sequences
    t1ce, _ = medio.load(os.path.join(src_path, file_name, 'T1c.nii.gz'))
    t1, _ = medio.load(os.path.join(src_path, file_name, 'T1.nii.gz'))
    t2, _ = medio.load(os.path.join(src_path, file_name, 'T2.nii.gz'))

    # Stack the volumes into a single 4D array
    volume = np.stack((t1ce, t1, t2), axis=0).astype(np.float32)

    # Crop and normalize the volume
    x_min, x_max, y_min, y_max, z_min, z_max = crop_volume(volume)
    normalized_volume = normalize_volume(volume[:, x_min:x_max, y_min:y_max, z_min:z_max])

    # Reorder the axes to match desired shape (x, y, z, channels)
    normalized_volume = normalized_volume.transpose(1, 2, 3, 0)

    # Save the preprocessed volume
    np.save(os.path.join(tar_path, f'{file_name}_vol.npy'), normalized_volume)


if __name__ == '__main__':
    root_directory = r'./NPC/Test/Test_resamp'
    target_directory_single = r"./NPC/Test/Test_npy"
    name_list = os.listdir(root_directory)

    # Use multiprocessing to process volumes concurrently
    with Pool(8) as pool:
        pool.starmap(preprocess_volume, zip(name_list, repeat(root_directory), repeat(target_directory_single)))
