import numpy as np
import torch
import pandas as pd
from pathlib import Path

def parse_chromosome_sizes(chrom_sizes_file: Path) -> dict[str, int]:
    """
    Parse chromosome sizes file into a dictionary: {chrom[str]: size[int]}
    """
    chrom_sizes = {}
    with open(chrom_sizes_file, 'r') as f:
        # for testing
        # for line in itertools.islice(f, 5, 7):
        #     chrom, size = line.strip().split()
        #     chrom_sizes[chrom] = int(size)
        for line in f:
            chrom, size = line.strip().split()
            chrom_sizes[chrom] = int(size)
    return chrom_sizes

def find_nan_runs(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find indices where arr is NaN, return as separate
    starts and ends NumPy arrays, respectively
    credit to ChatGPT
    """
    nan_indices = np.isnan(arr)

    # Compute differences between consecutive indices
    diff_indices = np.diff(nan_indices.astype(int))

    # Find indices where differences are 1 (indicating contiguous NaN values)
    start_indices = np.where(diff_indices == 1)[0] + 1
    end_indices = np.where(diff_indices == -1)[0]

    # Handle cases where NaN values are at the start or end of the array
    if nan_indices[0]:
        start_indices = np.insert(start_indices, 0, 0)
    if nan_indices[-1]:
        end_indices = np.append(end_indices, len(arr) - 1)

    # Return the starting and ending indices of NaN runs
    return start_indices, end_indices