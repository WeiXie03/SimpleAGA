import numpy as np, torch
from math import ceil
import pandas as pd
from pyBedGraph import BedGraph
import pyBigWig
import multiprocessing as mp, functools
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import itertools
import sys
from pathlib import Path
from .context import SimpleAGA
import pytest

def write_test_bigWig(chrom_sizes: pd.DataFrame, chrom_sig_vals: list[np.ndarray], bw_path: Path, chrom_sizes_path: Path) -> None:
    '''
    Writes a test bigWig file to bw_path, using
        - the chromosome sizes in `chrom_sizes_path`, and
        - the signal values in `chrom_sig_vals`.
    Format of `chrom_sizes`:
        columns: name <str>, size <int>, one row per chromosome
    Format of `chrom_sig_vals`: list of 1D numpy arrays of
        the signal values, one per chromosome.
    
    '''
    bw = pyBigWig.open(str(bw_path), "w")
    row_tuples = chrom_sizes.itertuples(index=False, name=None)
    bw.addHeader(row_tuples)

    for chrom_idx in range(len(chrom_sig_vals)):
        # ensure that the signal values are of the correct length
        assert(chrom_sig_vals[chrom_idx].shape[0] == chrom_sizes.iloc[chrom_idx, 1])
        bw.addEntries(chrom_sizes.iloc[chrom_idx, 0], 0, values=chrom_sig_vals[chrom_idx], span=1, step=1)
    
    bw.close()

class TestBinner:
    TEST_DATA_DIR = Path(".") / "test_data"

    def setup(self):
        self.TEST_DATA_DIR.mkdir(exist_ok=True)

    def one(self):
        chrom_sizes = pd.DataFrame([
            ("chr1", 10),
            ("chr2", 4),
            ("chr3", 7)], columns=["name", "size"])
        '''
        chr1: 0 1 0 1 0 1 0 1 0 1
        chr2: 1 0 1 0
        chr3: 0 1 0 1 0 1 0
        '''
        chrom_sig_vals = [
            np.tile(np.array([0,1]), 5),
            np.array([1,0]),
            np.array([0,1,0,1,0,1,0])]
        write_test_bigWig(chrom_sizes, chrom_sig_vals, self.TEST_DATA_DIR / "test1.bw", self.TEST_DATA_DIR / "test1.chrom.sizes")

        BIN_SIZE = 2 
        '''
        binned vals:
        chr1: 0.5 0.5 0.5 0.5 0.5
        chr2: 0.5 0.5
        chr3: 0.5 0.5 0.5 0
        '''
        bw_binner = SimpleAGA.BigWigsBinner([self.TEST_DATA_DIR / "test1.bw"], chrom_sizes, 1)
        binned_vals = bw_binner.load_bin_all_bws()

        assert(len(binned_vals) == 1)
        # chr1
        assert(binned_vals[0].size == 5)
        assert((binned_vals[0] == 0.5).all() == True)
        # chr2
        assert(binned_vals[1].size == 2)
        assert((binned_vals[1] == 0.5).all() == True)
        # chr3
        assert(binned_vals[2].size == 4)
        assert((binned_vals[2][:-1] == 0.5).all() == True)
        assert(binned_vals[2][-1] == 0)

if __name__ == "__main__":
    DATA_DIR = Path().resolve().parent.parent / "data"
    print(DATA_DIR.absolute())
    SIZES_FILE_PATH = DATA_DIR / "hg38.chrom.sizes"
    BIN_SIZE = 1000

    bw_paths = SimpleAGA.collect_bigWig_paths(DATA_DIR / "CD14-positive monocyte" / "H3K27ac")
    print(f"Found {len(bw_paths)} bigWigs")

    bw_binner = SimpleAGA.BigWigsBinner(bw_paths, SimpleAGA.parse_chromosome_sizes(SIZES_FILE_PATH), BIN_SIZE)
    bw_binner.load_all_bws_vals()
    bw_binned_tracks = bw_binner.bin_bigWigs()
    print(f"Loaded {bw_binner.raw_tracks.shape[0]} bigWigs into array of shape {bw_binner.raw_tracks.shape}")
    print("Sum of binned values:")
    for bw_idx in range(bw_binner.raw_tracks.shape[0]):
        print(f"\t{bw_paths[bw_idx]}: {np.nansum(bw_binned_tracks[bw_idx])}")
    bw_binner.save(bw_binned_tracks, DATA_DIR / "binned_vals.npy", DATA_DIR / "chrom_ranges.csv")
    print(f"Saved binned values to {DATA_DIR / 'binned_vals.npy'} and chromosome ranges to {DATA_DIR / 'chrom_ranges.csv'}")