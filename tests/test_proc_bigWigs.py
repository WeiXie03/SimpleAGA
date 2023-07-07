'''
bigWig binning test script using pytest
'''

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

def write_test_bigWig(chrom_sizes: pd.DataFrame, chrom_sig_vals: pd.DataFrame, bw_path: Path, chrom_sizes_path: Path) -> None:
    '''
    Writes a test bigWig file to bw_path, using
        - the chromosome sizes in `chrom_sizes_path`, and
        - the signal values in `chrom_sig_vals`.
    Format of `chrom_sizes`:
        columns: name <str>, size <int>, one row per chromosome
    Format of `chrom_sig_vals`: each row represents a signal value,
        which chromosome is included in a column
        columns: chrom_name <str>, start <int>, stop <int>, value <float>
    '''

    # TODO: Support arbitrary intervals (i.e. start and stop positions)
    # per value in chromosomes. Switch `chrom_sig_vals` to a single
    # signal values DataFrame, with columns: chrom, start, stop, value

    bw = pyBigWig.open(str(bw_path), "w")
    row_tuples = chrom_sizes.itertuples(index=False, name=None)
    bw.addHeader(list(row_tuples))

    # TODO? count and report # unmapped bases for debugging

    grouped_sigs = chrom_sig_vals.groupby("chrom_name")
    for chr_name, chr_group in grouped_sigs:
        print(f"Writing {chr_name}...")
        print(chr_group)

        df_cols_dict = chr_group.to_dict(orient="list")
        bw.addEntries((df_cols_dict["chrom_name"]), (df_cols_dict["start"]), ends=(df_cols_dict["stop"]), values=(df_cols_dict["value"]))
    bw.close()

    chrom_sizes.to_csv(chrom_sizes_path, sep='\t', index=False, header=False)

class TestBinner:
    TEST_DATA_DIR = Path(".") / "test_data"

    def test_setup(self):
        self.TEST_DATA_DIR.mkdir(exist_ok=True)

    def test_mono_alt0_1(self):
        chrom_sizes = pd.DataFrame([
            ("chr1", 10),
            ("chr2", 4),
            ("chr3", 7)], columns=["name", "size"])
        '''
        chr1: 0 1 0 1 0 1 0 1 0 1
        chr2: 1 0 1 0
        chr3: 0 1 0 1 0 1 0
        '''
        chr_names = [[name] * size for name, size in chrom_sizes.itertuples(index=False, name=None)]
        sig_vals = [
            list(np.tile(np.array([0,1], dtype=np.float32), 5)),
            list(np.tile(np.array([1,0], dtype=np.float32), 2)),
            list(np.array([0,1,0,1,0,1,0], dtype=np.float32))
        ]
        starts = [list(range(chr_size)) for chr_size in chrom_sizes["size"]]
        ends = [list(range(1, chr_size+1)) for chr_size in chrom_sizes["size"]]
        chrom_sig_vals = pd.DataFrame({
            "chrom_name": list(itertools.chain.from_iterable(chr_names)),
            "start": list(itertools.chain.from_iterable(starts)),
            "stop": list(itertools.chain.from_iterable(ends)),
            "value": list(itertools.chain.from_iterable(sig_vals))
        })

        write_test_bigWig(chrom_sizes, chrom_sig_vals, self.TEST_DATA_DIR / "test_mono_alt0_1.bw", self.TEST_DATA_DIR / "test_mono_alt0_1.chrom.sizes")

        BIN_SIZE = 2 
        '''
        binned vals:
        chr1: 0.5 0.5 0.5 0.5 0.5
        chr2: 0.5 0.5
        # NOTE: pyBigWig weird, bin from end,
        # leaving remainders at __beginning__
        chr3: 0 0.5 0.5 0.5
        '''
        bw_binner = SimpleAGA.BigWigsBinner([self.TEST_DATA_DIR / "test1.bw"], chrom_sizes, BIN_SIZE)
        binned_vals = bw_binner.load_bin_all_bws()

        assert(len(binned_vals) == 1)
        print(f"\t{binned_vals}")
        # chr1
        assert(binned_vals[0][0].size == 5)
        assert((binned_vals[0][0] == 0.5).all() == True)
        # chr2
        assert(binned_vals[0][1].size == 2)
        assert((binned_vals[0][1] == 0.5).all() == True)
        # chr3
        assert(binned_vals[0][2].size == 4)
        assert((binned_vals[0][2][1:] == 0.5).all() == True)
        assert(binned_vals[0][2][0] == 0)

    def test_sequential_missing(self):
        chrom_sizes = pd.DataFrame([
            ("chr1", 10),
            ("chr2", 4),
            ("chr3", 7)], columns=["name", "size"])
        '''
        # `-` denotes missing
        chr1: 0 - 0 1 2 3 - - 0 1
        chr2: 0 1 2 3
        chr3: - 0 - 0 1 2 -
        '''
        chr_names = [[name] * size for name, size in chrom_sizes.itertuples(index=False, name=None)]
        sig_vals = [
            [0, np.nan, 0, 1, 2, 3, np.nan, np.nan, 0, 1],
            [0, 1, 2, 3],
            [np.nan, 0, np.nan, 0, 1, 2, np.nan]
        ]
        starts = [list(range(chr_size)) for chr_size in chrom_sizes["size"]]
        ends = [list(range(1, chr_size+1)) for chr_size in chrom_sizes["size"]]
        chrom_sig_vals = pd.DataFrame({
            "chrom_name": list(itertools.chain.from_iterable(chr_names)),
            "start": list(itertools.chain.from_iterable(starts)),
            "stop": list(itertools.chain.from_iterable(ends)),
            "value": list(itertools.chain.from_iterable(sig_vals))
        })

        write_test_bigWig(chrom_sizes, chrom_sig_vals, self.TEST_DATA_DIR / "test_sequential_missing.bw", self.TEST_DATA_DIR / "test_sequential_missing.chrom.sizes")

        BIN_SIZE = 2 
        '''
        binned vals:
        chr1: - 0.5 2.5 - 0.5
        chr2: 0.5 2.5
        # NOTE: pyBigWig weird, bin from end,
        # leaving remainders at __beginning__
        chr3: - - 0.5 -
        '''
        bw_binner = SimpleAGA.BigWigsBinner([self.TEST_DATA_DIR / "test_sequential_missing.bw"], chrom_sizes, BIN_SIZE)
        binned_vals = bw_binner.load_bin_all_bws()

        assert(len(binned_vals) == 1)
        print(f"\t{binned_vals}")
        # chr1
        assert(binned_vals[0][0].size == 5)
        assert(np.isnan(binned_vals[0][0][0]) and np.isnan(binned_vals[0][0][3]))
        assert(binned_vals[0][0][1] == 0.5 and binned_vals[0][0][4] == 0.5)
        assert(binned_vals[0][0][2] == 2.5)
        # chr2
        assert(binned_vals[0][1].size == 2)
        assert(binned_vals[0][1][0] == 0.5 and binned_vals[0][1][1] == 2.5)
        # chr3
        assert(binned_vals[0][2].size == 4)
        assert(binned_vals[0][2][2] == 0.5)
        assert(np.nansum(binned_vals[0][2]) == 0.5)

        missings_tbl = pd.DataFrame(bw_binner.missing_bins)
        print("Missing singal values:")
        print(missings_tbl)

'''
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
'''