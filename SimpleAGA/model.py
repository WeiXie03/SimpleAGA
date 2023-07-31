'''
The SAGA, a Hidden Markov Model
'''

import numpy as np
import torch
import pandas as pd
from pyBedGraph import BedGraph
import multiprocessing as mp
import functools
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import itertools
import os
from enum import Enum
from pathlib import Path
import pomegranate, pomegranate.hmm
import _util

class HandleMissingStrategy(Enum):
    OMIT = "omit"
    MARGINALIZE = "marginalize"

def omit_missing(chrom_arr: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
    """
    Omit positions at which _any_ of the tracks are missing values from all_binneds
    Returns:
        - a new tensor with the missing positions omitted, and
        - a tensor of the indices of the omitted positions
    For a single chromosome (represented by one 2D tensor, each row a bigWig track)
    """
    # Find columns with NaN values
    # "Any" _through_ columns, reducing the __0__th dimension
    nan_cols = torch.any(torch.isnan(chrom_arr), 0)
    # Convert from bool mask to indices
    nan_col_idx = torch.where(nan_cols)[0]
    # Remove columns with NaN values
    return chrom_arr[:, ~nan_cols], nan_col_idx

def load_binned_chrom(chr_name: str, data_dir: Path) -> tuple[str, torch.Tensor]:
    """
    Load binned data for chromosome chr_name from its torch .pt file in data_dir,
    assuming each .pt file is named <chr_name>.pt
    Returns a tuple of the chromosome name and the loaded tensor
    """
    try:
        data = torch.load(data_dir/(chr_name+".pt"))
    except FileNotFoundError:
        print(f"{chr_name}.pt not found")
    finally:
        return (chr_name, data)

def load_all_chrom_tensors(chr_names: [str], data_dir: Path, n_threads=None) -> dict[str, torch.Tensor]:
    """
    """
    if not isinstance(n_threads, int):
        if n_threads != None:
            print("Error: n_threads must be an int")
        n_threads = len(chr_names)
    
    print(f"Loading {len(chr_names)} chromosomes from {data_dir} with {n_threads} threads...")

    with ThreadPoolExecutor(n_threads) as pool:
        fixed_load = functools.partial(load_binned_chrom, data_dir=data_dir)
        itemsL = pool.map(fixed_load, chr_names)
        return dict(itemsL)

class omitterSAGA:
    """
    A SAGA that omits missing values
    """
    def __init__(self, all_binneds: dict[str, torch.Tensor], n_procs: int, handle_missing=HandleMissingStrategy.OMIT):
        """
        all_binneds: a dictionary of chromosome names to 2D tensors, each row a bigWig track
        n_procs: number of processes to use for parallelization
        handle_missing: how to handle missing values
        """

        # Load in the binned data (tensors)
        self.all_binneds = all_binneds

        self.n_procs = n_procs
        with mp.Pool(n_procs) as mpool:
            self.observations, _ = mpool.map(omit_missing, all_binneds)
            # self.missing_rngs_tbl = pd.DataFrame()

if __name__ == "__main__":
    # DATA_DIR = Path("../../data")
    # chrom_sizes_path = DATA_DIR / "hg38.chrom.sizes"
    DATA_DIR = Path("../tests/test_data/test_mono_alt0_1")
    chrom_sizes_path = DATA_DIR / "mono_alt0_1.chrom.sizes"

    chrom_sizes = _util.parse_chromosome_sizes(chrom_sizes_path)
    chrom_sizes = pd.DataFrame(chrom_sizes.items(), columns=["name", "size"])

    # Load in the binned data (tensors)
    all_binneds = load_all_chrom_tensors(chrom_sizes["name"], DATA_DIR/"binned")
    print(all_binneds)