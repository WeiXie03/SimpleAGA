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

def slice_chrom_bins_genomic_coords(chrom_tensor: np.ndarray, coords_df: pd.DataFrame, bin_size: int) -> np.ndarray:
    """
    Slice out of the binned data for the chromosome the intervals specified by the given genomic coordinates.

    Args
    ----
    chrom_tensor: The tensor for the chromosome.
    coords_df: A DataFrame with columns "start", "end" for the genomic coordinates of the bins. Ignores other columns.
    bin_size: The size of the bins in the data.
    """
    chrom_bins = coords_df.apply(lambda row: chrom_tensor[row["start"]:row["end"]+1], axis=1)
    return np.stack(chrom_bins.values)

def slice_chroms_bins_genomic_coords(chroms_tensors: dict[str, np.ndarray], coords_df: pd.DataFrame, bin_size: int, n_procs: int = None) -> dict[str, np.ndarray]:
    """
    Slice out of the binned data for each chromosome the intervals specified by the given genomic coordinates.

    Args
    ----
    chroms_tensors: A dictionary of the chromosome names and their corresponding tensors.
    coords_df: A DataFrame with columns "chr", "start", "end" for the genomic coordinates of the bins.
    bin_size: The size of the bins in the data.
    """
    if n_procs == None:
        n_procs = mp.cpu_count()

    with mp.Pool(mp.cpu_count()) as pool:
        full_tensors_coords_pairs = [(chroms_tensors[chrom], coords_df[coords_df["chr"] == chrom], bin_size) for chrom in chroms_tensors.keys()]
        tensors_slices = pool.starmap(slice_chrom_bins_genomic_coords, full_tensors_coords_pairs)
        chroms_bins = dict(zip(chroms_tensors.keys(), tensors_slices))
    return chroms_bins

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

def mp_arrays_lens(arrays: list[np.ndarray], n_procs: int) -> np.ndarray:
    """
    Return the lengths of each array in arrays
    """
    with mp.Pool(n_procs) as pool:
        return np.array(pool.map(len, arrays))

def slice_rand_subseq_idx(arr_len: int, subseq_len: int, rand_rng: np.random.BitGenerator = None) -> tuple[int,int]:
    """
    Return a tuple of [start, end) for a random contiguous subsequence of length subseq_len
    """
    if 0 > subseq_len or subseq_len > arr_len:
        raise ValueError("subseq_len must be in [0, arr_len]")
    
    if rand_rng == None:
        rand_rng = np.random.default_rng()

    start = rand_rng.integers(0, arr_len - subseq_len + 1, endpoint=True)
    # print(" ", start, "->", start + subseq_len, " ")
    end = start + subseq_len
    return (start, end)

def slice_rand_subseqs_idx(arr_lens: np.ndarray, subseq_len: int, rand_rng: np.random.BitGenerator = None) -> list[tuple[int,int]]:
    """
    Return a list of tuples of [start, end) for a random contiguous subsequence of length subseq_len for each array in arr_lens
    """
    if rand_rng == None:
        rand_rng = np.random.default_rng()
    if np.any(arr_lens < subseq_len):
        raise ValueError("subseq_len must be less than every array length")
    
    idxs = []
    for arr_len in arr_lens:
        start = rand_rng.integers(0, arr_len - subseq_len + 1, endpoint=True)
        # ensure no overlap and that subsequence will not extend past the end of the array
        while (any([start < s and s < start + subseq_len for s, _ in idxs]) or
               start + subseq_len > arr_len):
            start = rand_rng.integers(0, arr_len - subseq_len + 1, endpoint=True)
        idxs.append((start, start + subseq_len))

    return idxs
    
def sample_minibatches(arrays: list[np.ndarray], frac: float, subseq_len: int = None,
                       lens: np.ndarray = None,
                       rand_gen: np.random.BitGenerator = None, n_proc: int = None) -> list[np.ndarray]:
    """
    _WARNING_: The default of this function when subseq_len is not given is broken. If more than one subsequence per array is desired, this will not sample enough to cover frac of the genome.
    For each array in arrays, return a random contiguous subsequence of length frac*len(array)
    If `subsample_len` is given, samples `n` random subsequences each of length
    `subsample_len`, where `n`*`subsample_len` = `frac` * $sum_{seq} |seq|$.
    Otherwise, samples one random subsequence from each given sequence of length
    `frac`*length(seq).
    """

    if n_proc == None:
        n_proc = mp.cpu_count()

    if lens is None:
        lens = mp_arrays_lens(arrays, n_proc)
    
    if subseq_len == None:
        with mp.Pool(n_proc) as pool:
            subseq_idx = pool.starmap(slice_rand_subseq_idx, [(l, int(frac * l)) for l in lens])
            # for _, (start, end) in zip(arrays, subseq_idx):
            #     print(" ", start, "->", end, " ")
            return [arr[start:end] for arr, (start, end) in zip(arrays, subseq_idx)]
    else:
        total_len = sum(lens)
        # better to sample a little more than less
        n_samples = ceil(frac*total_len / subseq_len)

        # n_samples must not be so small such that a subsample longer than the entire shortest interval
        if subseq_len > min(lens):
            raise ValueError("n_samples too large, subsample longer than the whole shortest interval")
        # n_samples must not be so large such that a subsample shorter than 1
        if subseq_len < 1:
            raise ValueError("n_samples too small, subsample shorter than 1")

        # each subsample is of length frac * sum{len(array)} / n_samples
        # sample n_samples of these random contiguous subsequences
        rand_rng = np.random.default_rng()
        chosen_arrays_idx = rand_rng.choice(len(arrays), n_samples, replace=True)
        subseq_idx = slice_rand_subseqs_idx(lens[chosen_arrays_idx], subseq_len, rand_rng)
        return [arrays[i][start:end] for i, (start, end) in zip(chosen_arrays_idx, subseq_idx)]