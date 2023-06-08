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
import argparse

def init_argparser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("data_dir", type=Path, help="Directory containing all bigWigs to be parsed, including within subdirectories.")
    parser.add_argument("bin_size", type=int, help="Size of bins to downsample the signal tracks to in base pairs. Also called resolution.")

def parse_chromosome_sizes(chrom_sizes_file: Path) -> dict[str, int]:
    """
    Parse chromosome sizes file into a dictionary: {chrom[str]: size[int]}
    """
    chrom_sizes = {}
    with open(chrom_sizes_file, 'r') as f:
        # for line in f:
        # TEMPORARY for testing
        for line in itertools.islice(f, 5, 7):
            chrom, size = line.strip().split()
            chrom_sizes[chrom] = int(size)
    return chrom_sizes

def collect_bigWig_paths(bgs_root: Path) -> list[Path]:
    """
    Return a list of paths of all bigWigs in `bgs_root` directory
    """
    if not bgs_root.is_dir():
        raise ValueError(f"{bgs_root} is not a directory")
    # TEMPORARY for testing
    return list(bgs_root.rglob("*.bigWig"))

# TODO: argparse -> genomedata-style {--track <track_name> <track_file>}_i, i = 1 to n

def open_bigwigs(bigwig_paths: list[Path], parallel: bool) -> list:
    if parallel:
        n_threads = len(bigwig_paths)
    else:
        n_threads = 1
    print(f"Opening {len(bigwig_paths)} bigwigs in {n_threads} threads", flush=True)

    bw_paths_strs = [str(path.absolute()) for path in bigwig_paths]
    bw_objs = None
    with ThreadPoolExecutor(max_workers=n_threads) as thr_pool:
        bw_objs = thr_pool.map(pyBigWig.open, bw_paths_strs)
    # for bw_path_str in bw_paths_strs:
    #     bw_obj = pyBigWig.open(bw_path_str)
    #     if bw_objs is None:
    #         bw_objs = [bw_obj]
    #     else:
    #         bw_objs.append(bw_obj)
    return bw_objs

def close_bigwigs(bigwig_objs: list) -> None:
    for bw in bigwig_objs:
        bw.close()

class BigWigsBinner:
    def __init__(self, bigwig_paths: list[Path], chrom_sizes: dict[str, int], bin_size: int, parallel=True):
        self.bin_size = bin_size
        self.parallel = parallel

        chrom_sizes_vals = list(chrom_sizes.values())
        # NOTE: currently only support all bigWigs same assembly => same chrom sizes
        # [If items(), keys(), values(), ... are called with
        # no intervening modifications to the dictionary,
        # the lists will directly correspond.](https://stackoverflow.com/a/835430)
        self.chrom_ranges = pd.DataFrame({
            "chrom_name": list(chrom_sizes.keys()),
            "start_pos": np.cumsum([0] + chrom_sizes_vals)[:-1],
            "len": chrom_sizes_vals
        })

        # will first just copy out everything in bigWigs into memory
        self.total_track_len = self.chrom_ranges["len"].sum()
        # for easy NumPy averaging, will reshape to a subarray per bin
        # if don't divide evenly into bins, in order to allocate 3D array,
        # will need to manually ignore last few values when averaging
        total_alloc_len = ceil(self.total_track_len / float(bin_size)) * bin_size
        self.raw_tracks = np.empty((len(bigwig_paths), total_alloc_len), dtype=np.float64)
        print(f"Allocated {self.raw_tracks.nbytes / 1e9} GB for raw tracks, a {self.raw_tracks.shape} {type(self.raw_tracks)}", flush=True)
        # have to cut last bin short if not divisible by bin_size
        self.n_excess_poss = self.total_track_len % bin_size

        # ACTUALLY, just provide a public method that returns binned values
        # # will store the binned values for all bigWigs
        # # in one giant array, a row for each bigWig
        # self.binned_vals = np.empty((len(bigwig_paths), self.n_bins), dtype=torch.float64)
    
        # assign space for each bigWig in binned values
        # identify each bigWig by its filename stem
        self.bigwigs_tbl = pd.DataFrame({
            "path": bigwig_paths,
            "bw_obj": open_bigwigs(bigwig_paths, parallel),
        })

    def __del__(self):
        close_bigwigs(self.bigwigs_tbl["bw_obj"])
        del self.raw_tracks
        del self.chrom_ranges
        del self.bigwigs_tbl

    def get_chrom_range_by_idx(self, chrom_idx: int) -> tuple[str, int, int]:
        chrom_name = self.chrom_ranges.iloc[chrom_idx]["chrom_name"]
        chrom_start = self.chrom_ranges.iloc[chrom_idx]["start_pos"]
        chrom_end = chrom_start + self.chrom_ranges.iloc[chrom_idx]["len"] - 1
        return (chrom_name, chrom_start, chrom_end)

    def load_bw_vals(self, bigwig, bw_idx: int) -> None:
        """
        "Loads" all the raw signal values into the
        `bw_idx`th row of the `raw_tracks` array
        """
        for chrom_idx in range(len(self.chrom_ranges)):
            chrom_name = self.chrom_ranges.iloc[chrom_idx]["chrom_name"]
            chrom_start = self.chrom_ranges.iloc[chrom_idx]["start_pos"]
            chrom_len = self.chrom_ranges.iloc[chrom_idx]["len"]
            chrom_end = chrom_start + chrom_len - 1

            chrom_vals = bigwig.values(chrom_name, 0, chrom_len-1, numpy=True)

            print(f"Loading {chrom_name} into `raw_tracks`, {np.count_nonzero(np.isnan(chrom_vals))} bases without signal values, sum = {np.nansum(chrom_vals)}", flush=True)
            self.raw_tracks[bw_idx, chrom_start:chrom_end] = chrom_vals
            print(f"Now in `raw_tracks[{bw_idx}, {chrom_start}:{chrom_end}]`, sum = {np.nansum(self.raw_tracks[bw_idx, chrom_start:chrom_end])}", flush=True)
    
    def load_all_bws_vals(self) -> None:
        if self.parallel:
            n_threads = len(self.bigwigs_tbl.index)
        else:
            n_threads = 1
        print(f"Loading {n_threads} bigWigs' signal values into NumPy arrays in {n_threads} threads", flush=True)

        # for (bw_idx, bw_obj) in enumerate(self.bigwigs_tbl["bw_obj"]):
        #     self.load_bw_vals(bw_obj, bw_idx)
        with ThreadPoolExecutor(max_workers=n_threads) as thr_pool:
            futures = []
            for (bw_idx, bw_obj) in enumerate(self.bigwigs_tbl["bw_obj"]):
                futures.append(thr_pool.submit(self.load_bw_vals, bw_obj, bw_idx))
            # for future in futures:
            #     print(f"Future {future} running? {future.running()}", flush=True)
        
        print(f"Done loading bigWigs' signal values into NumPy array `raw_tracks`, shape {self.raw_tracks.shape}", flush=True)

    # def bin_chrom(self, bw_idx: int, chrom_idx: int) -> None:
    #     (chrom_name, chrom_start, chrom_end) = self.get_chrom_range_by_idx(chrom_idx)
    #     self.binned_vals[bw_idx, chrom_start:chrom_end] = self.raw_tracks[bw_idx, chrom_start:chrom_end].mean(axis=1)

    # def bin_one(self, bigwig, bw_idx: int) -> None:
    #     """
    #     Stores the binned values for all chromosomes in `raw_tracks`
    #     """
    #     # NOTE: unfortunately can't parallelize because
    #     # pyBigWig.bigWigFiles not picklable, and goddamn
    #     # Python just has to pickle object to share across procs
    #     for (chrom_name, chrom_start, chrom_end) in self.chrom_ranges.apply(lambda row: self.get_chrom_range_by_idx(row.name), axis=1):
    #         self.binned_vals[bw_idx, chrom_start:chrom_end] = self.bin_chrom(bigwig, chrom_name)

    def bin_bigWigs(self) -> np.ndarray:
        """
        Bin all bigWigs into bins of size bin_size simultaneously,
        return as a NumPy array, shape 
        """
        # Now that loaded, CPU-bound
        # Unfortunately, pyBigWig.bigWigFiles not picklable
        # with mp.Pool() as pool:
        #     bin_vals = pool.map(functools.partial(bin_bigWig, bin_size=bin_size), bw_objs)
        n_bws = len(self.bigwigs_tbl.index)
        # mean(sum and div) through the dimension along which
        # all values for each individual bin reside
        # total length of genome divided evenly into bins of size `bin_size`
        print(f"Length of `raw_tracks`: {self.raw_tracks.size}", flush=True)
        print(f"Shape of `raw_tracks`: {self.raw_tracks.shape}", flush=True)
        n_bins_alloc_space = int(self.raw_tracks.shape[1] / self.bin_size)
        print(f"Dividing genome into {n_bins_alloc_space} bins of size {self.bin_size}", flush=True)
        vals_in_bins = self.raw_tracks.reshape((-1, n_bins_alloc_space, self.bin_size))
        print(f"Averaging through an array of shape {vals_in_bins.shape}", flush=True)
        binned_vals = vals_in_bins.mean(axis=2)
        print(f"Averages (binned values) in a {binned_vals.shape} array.", flush=True)
        # deal with junk remainder positions from uneven division into bins
        # print("Last bin of every bigWig:", vals_in_bins[:, -1, -self.n_excess_poss:])
        binned_vals[:,-1] = vals_in_bins[:, -1, :-self.n_excess_poss].mean(axis=1)
        return binned_vals
    
    def save(self, binned_vals: np.ndarray, np_save_path: Path, chrom_ranges_save_path: Path) -> None:
        """
        Save the binned values to a NumPy array
        """
        np.save(np_save_path, binned_vals)
        self.chrom_ranges.to_csv(chrom_ranges_save_path, index=False)

if __name__ == "__main__":
    DATA_DIR = Path().resolve().parent.parent / "data"
    print(DATA_DIR.absolute())
    SIZES_FILE_PATH = DATA_DIR / "hg38.chrom.sizes"
    BIN_SIZE = 1000

    bw_paths = collect_bigWig_paths(DATA_DIR / "CD14-positive monocyte" / "H3K27ac")
    print(f"Found {len(bw_paths)} bigWigs")

    bw_binner = BigWigsBinner(bw_paths, parse_chromosome_sizes(SIZES_FILE_PATH), BIN_SIZE)
    bw_binner.load_all_bws_vals()
    bw_binned_tracks = bw_binner.bin_bigWigs()
    print(f"Loaded {bw_binner.raw_tracks.shape[0]} bigWigs into array of shape {bw_binner.raw_tracks.shape}")
    print("Sum of binned values:")
    for bw_idx in range(bw_binner.raw_tracks.shape[0]):
        print(f"\t{bw_paths[bw_idx]}: {np.nansum(bw_binned_tracks[bw_idx])}")
    bw_binner.save(bw_binned_tracks, DATA_DIR / "binned_vals.npy", DATA_DIR / "chrom_ranges.csv")
    print(f"Saved binned values to {DATA_DIR / 'binned_vals.npy'} and chromosome ranges to {DATA_DIR / 'chrom_ranges.csv'}")