import os
import numpy as np
import medpy.io as medio  # A module from the MedPy library for medical image processing
from multiprocessing import Pool
from itertools import repeat


def ensure_minimum_dimension(xmin, xmax, min_size=128):
    """Ensure the dimension size is at least 'min_size'."""
    if xmax - xmin < min_size:
        print('#' * 100)
        padding = (min_size - (xmax - xmin)) // 2
        xmax += padding + 1
        xmin -= padding
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
    # Ensure the input volume is 3D (if it's 4D, collapse it into a single 3D volume)
    if len(volume.shape) == 4:
        volume = np.amax(volume, axis=0)

    assert len(volume.shape) == 3, "Expected a 3D volume"

    non_zero_coords = np.where(volume != 0)

    # Calculate cropping boundaries for each axis
    x_min, x_max = np.amin(non_zero_coords[0]), np.amax(non_zero_coords[0])
    y_min, y_max = np.amin(non_zero_coords[1]), np.amax(non_zero_coords[1])
    z_min, z_max = np.amin(non_zero_coords[2]), np.amax(non_zero_coords[2])

    # Ensure minimum dimension size of 128
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
    for k in range(volume.shape[0]):  # Iterate over each channel (T1c, T1, T2)
        channel = volume[k, ...]
        non_zero_values = channel[mask]
        # Normalize per channel based on non-zero values
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
    output_file = os.path.join(tar_path, f'{file_name}_vol.npy')

    # Skip if preprocessed file already exists
    if os.path.exists(output_file):
        return

    # Load the MRI sequences, handling potential errors in file loading
    try:
        t1ce, _ = medio.load(os.path.join(src_path, file_name, 'T1c.nii.gz'))
        t1, _ = medio.load(os.path.join(src_path, file_name, 'T1.nii.gz'))
        t2, _ = medio.load(os.path.join(src_path, file_name, 'T2.nii.gz'))
    except Exception as e:
        print(f"Error loading files for {file_name}: {e}")
        return

    # Stack the volumes into a single 4D array
    volume = np.stack((t1ce, t1, t2), axis=0).astype(np.float32)

    # Crop and normalize the volume
    x_min, x_max, y_min, y_max, z_min, z_max = crop_volume(volume)
    normalized_volume = normalize_volume(volume[:, x_min:x_max, y_min:y_max, z_min:z_max])

    # Reorder the axes to match desired shape (x, y, z, channels)
    normalized_volume = normalized_volume.transpose(1, 2, 3, 0)

    # Save the preprocessed volume
    try:
        np.save(output_file, normalized_volume)
    except Exception as e:
        print(f"Error saving file {output_file}: {e}")


def process_files_in_parallel(root_directory, target_directory, num_workers=8):
    """
    Process multiple files in parallel using multiprocessing.

    Args:
        root_directory: The source directory containing subdirectories for each file.
        target_directory: The target directory to save the processed files.
        num_workers: The number of parallel workers (default: 8).
    """
    name_list = os.listdir(root_directory)

    # Use multiprocessing to process volumes concurrently
    with Pool(num_workers) as pool:
        pool.starmap(preprocess_volume, zip(name_list, repeat(root_directory), repeat(target_directory)))


if __name__ == '__main__':
    root_directory = r'./NPC/Test/Test_resamp'
    target_directory_single = r"./NPC/Test/Test_npy"

    # Call the parallel processing function
    process_files_in_parallel(root_directory, target_directory_single)
