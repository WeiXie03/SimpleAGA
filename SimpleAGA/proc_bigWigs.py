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
import pickle
import argparse

def parse_chromosome_sizes(chrom_sizes_file: Path) -> dict[str, int]:
    """
    Parse chromosome sizes file into a dictionary: {chrom[str]: size[int]}
    """
    chrom_sizes = {}
    with open(chrom_sizes_file, 'r') as f:
        # for testing
        for line in itertools.islice(f, 5, 7):
            chrom, size = line.strip().split()
            chrom_sizes[chrom] = int(size)
        # for line in f:
        #     chrom, size = line.strip().split()
        #     chrom_sizes[chrom] = int(size)
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

        # NOTE: currently only support all bigWigs same assembly => same chrom sizes
        self.chrom_sizes = pd.DataFrame({
            "name": chrom_sizes.keys(),
            "size": chrom_sizes.values(),
            "n_bins": [ceil(size / bin_size) for size in chrom_sizes.values()],
        })
        self.bigwigs_tbl = pd.DataFrame({
            "path": bigwig_paths,
            "bw_obj": open_bigwigs(bigwig_paths, parallel),
        })

        # store everything in a 2D list of numpy arrays,
        # where each row is a bigWig and each column is a chromosome,
        # one array per chromosome
        self.binned_vals = [[] for _ in range(len(self.bigwigs_tbl.index))]

    def __del__(self):
        close_bigwigs(self.bigwigs_tbl["bw_obj"])
        del self.chrom_sizes
        del self.bigwigs_tbl

    def load_bin_bw(self, bigwig, bw_idx: int) -> None:
        """
        "Loads" and bins all the signal values into the
        `bw_idx`th row of the `binned_vals` 2D list
        """
        for chr_name, chr_len, chr_bins in self.chrom_sizes.itertuples(index=False, name=None):
            print(f"Loading {chr_name} of {bw_idx}th bigWig", flush=True)
            self.binned_vals[bw_idx].append(bigwig.stats(chr_name, nBins=chr_bins, type="mean", numpy=True))
            print(f"Now in `binned_vals[{bw_idx}]`, sum = {np.nansum(self.binned_vals[bw_idx][-1])}", flush=True)
    
    def load_bin_all_bws(self) -> list[list[np.ndarray]]:
        if self.parallel:
            n_threads = len(self.bigwigs_tbl.index)
        else:
            n_threads = 1
        print(f"Loading {n_threads} bigWigs' signal values into NumPy arrays in {n_threads} threads", flush=True)

        # for (bw_idx, bw_obj) in enumerate(self.bigwigs_tbl["bw_obj"]):
        #     self.load_bin_bw(bw_obj, bw_idx)
        with ThreadPoolExecutor(max_workers=n_threads) as thr_pool:
            futures = []
            for (bw_idx, bw_obj) in enumerate(self.bigwigs_tbl["bw_obj"]):
                futures.append(thr_pool.submit(self.load_bin_bw, bw_obj, bw_idx))
            # for future in futures:
            #     print(f"Future {future} running? {future.running()}", flush=True)
        
        print(f"Done loading bigWigs' signal values into 2D list of NumPy arrays, {len(self.binned_vals)} rows", flush=True)
        return self.binned_vals
    
    def save(self, binned_vals: list[list[np.ndarray]], save_path: Path, chroms_tbl_save_path: Path) -> None:
        """
        Save the binned values to a NumPy array
        """
        #np.save(np_save_path, binned_vals)
        pickle.dump(binned_vals, open(save_path, "wb"))
        self.chrom_sizes.to_csv(chroms_tbl_save_path, index=False)

def init_argparser(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    - downloads dir
    - dir for segway runs
    - dir for final reports
    """
    parser.add_argument("data_dir", type=Path, help="Path to directory of data (contains subdirectories for cell types).")
    parser.add_argument("resolution", type=int, help="Requested resolution (i.e. bin size) in base pairs.")
    parser.add_argument("--chrom-sizes", type=Path, help="Path to `.sizes` file to use. If not specified, will use `hg38.chrom.sizes` in `data_dir` directory")
    args = parser.parse_args()
    if args.chrom_sizes is None:
        args.chrom_sizes = args.data_dir / "hg38.chrom.sizes"
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bins bigWig files into NumPy arrays.\nSpecifically, a 2D list of NumPy arrays, where each row is a bigWig and each column is a chromosome. Saves the 2D list as a pickle and a table of chromosomes information used, namely name, length (bp) and number of bins under given bin size (i.e. resolution)."
    )
    args = init_argparser(parser)

    bw_paths = collect_bigWig_paths(args.data_dir / "CD14-positive monocyte" / "H3K27ac")
    print(f"Found {len(bw_paths)} bigWigs")

    bw_binner = BigWigsBinner(bw_paths, parse_chromosome_sizes(args.chrom_sizes), args.resolution)
    bw_binned_tracks = bw_binner.load_bin_all_bws()
    print("Sum of binned values:")
    for bw_idx in bw_binner.bigwigs_tbl.index:
        print(f"\t{bw_paths[bw_idx]}: {sum([np.nansum(bw_binned_tracks[bw_idx][chr_idx]) for chr_idx in range(len(bw_binned_tracks[bw_idx]))])}")
    bw_binner.save(bw_binned_tracks, args.data_dir / "binned_vals.npy", args.data_dir / "chrom_ranges.csv")
    print(f"Saved binned values to {args.data_dir / 'binned_vals.npy'} and chromosome ranges to {args.data_dir / 'chrom_ranges.csv'}")