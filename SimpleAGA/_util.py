from math import ceil
import numpy as np
import torch
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import functools
import threading
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

def gen_chrom_intervs_tbl(chrom: str, procd_data: np.ndarray, missing_idx: np.ndarray, 
                           chrom_sizes: pd.Series, bin_size: int) -> pd.DataFrame:
    # TODO: move model.predict_proba() to all chroms function by using lengths arg
    """
    Generate a DataFrame with columns for the original intervals of the
    bins for the given chromosome.
    To be used for generating the parsed posteriors table.

    Args
    ----
    chrom_sizes: A Series of the sizes of the chromosomes in
    the data to train on. Must have:
        - An index of the chromosome names.
        - A column "size": The sizes of the chromosome.
    """
    chrom_size = chrom_sizes[chrom]

    n_bins = ceil(chrom_size/bin_size)
    print(f"Assuming ceil({chrom_size}/{bin_size}) = {n_bins} bins.")

    # bigWigs2tensors leaves unevenly divided
    # remainder values in the last bin.
    bin_rem = chrom_size % bin_size
    starts = np.arange(0, chrom_size, bin_size, dtype=np.int32)
    ends = starts + (bin_size - 1)
    # After entire ends column is generated, correct last
    ends[-1] = starts[-1] + bin_rem

    # Convert indices to a mask
    avail_mask = np.ones((n_bins,), dtype=bool)
    avail_mask[missing_idx] = False

    return pd.DataFrame({
        "chr": chrom,
        "start": starts[avail_mask],
        "end": ends[avail_mask]
    })

def mp_arrays_lens(arrays: list[np.ndarray], n_procs: int) -> list[int]:
    """
    Return the lengths of each array in arrays
    """
    with mp.Pool(n_procs) as pool:
        return list(pool.map(len, arrays))