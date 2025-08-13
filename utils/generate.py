import os
import copy
import numpy as np


def normalize_and_tile(f, H, W, T):
    """Normalize and tile the input 3D array to match desired format"""
    f = ((f - np.min(f)) * 255.0 / (np.max(f) - np.min(f))).astype(np.uint8)
    return np.tile(f[:H, :W, np.newaxis, :T], (1, 1, 3, 1))


def create_empty_fig(H, W, T, mask_condition):
    """Create an empty figure where the mask condition is set to 255"""
    empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
    empty_fig[np.where(mask_condition)] = 255
    return empty_fig


def generate_snapshot(x, H, W, T, output, target_cpu, gap_width=2):
    """Generate snapshot with highlighted regions based on output and target comparisons"""

    # Step 1: Normalize and tile input channels
    f1 = normalize_and_tile(x[0], H, W, T)
    f2 = normalize_and_tile(x[1], H, W, T)
    f3 = normalize_and_tile(x[2], H, W, T)
    f4 = normalize_and_tile(x[3], H, W, T)

    # Step 2: Initialize snapshot image with the required shape
    Snapshot_img2 = np.zeros(shape=(H, H * 4 + gap_width * 3, 3, T), dtype=np.uint8)

    # Step 3: Add gaps between columns
    Snapshot_img2[:, W:W + gap_width, :] = 255
    Snapshot_img2[:, 2 * W + gap_width:2 * W + 2 * gap_width, :] = 255
    Snapshot_img2[:, 3 * W + 2 * gap_width:3 * W + 3 * gap_width, :] = 255

    # Step 4: Create empty figures for different conditions (output vs target comparison)
    for cls in [1, 2, 3]:
        for idx, mask_condition in enumerate([output == cls, target_cpu == cls, (target_cpu == cls) & (output != cls),
                                              (target_cpu != cls) & (output == cls)]):
            empty_fig = create_empty_fig(H, W, T, mask_condition)
            Snapshot_img2[:, idx * (W + gap_width): (idx + 1) * (W + gap_width), idx, :] = empty_fig

    # Step 5: Create gaps between rows
    gap_horizon = np.ones(shape=(gap_width, W * 4 + 3 * gap_width, 3, T), dtype=np.uint8) * 255
    gap_vetical = np.ones(shape=(H, gap_width, 3, T), dtype=np.uint8) * 255

    # Step 6: Concatenate images horizontally and vertically
    Snapshot_img1 = np.concatenate((f1, gap_vetical, f2, gap_vetical, f3, gap_vetical, f4), axis=1)
    Snapshot_img = np.concatenate((Snapshot_img1, gap_horizon, Snapshot_img2), axis=0)

    return Snapshot_img
