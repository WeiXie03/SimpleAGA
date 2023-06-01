import numpy as np, torch
import pandas as pd
from pyBedGraph import BedGraph
import pyBigWig
import multiprocessing as mp, functools
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import itertools
from math import ceil
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
        for line in f:
            chrom, size = line.strip().split()
            # print(chrom, size)
            chrom_sizes[chrom] = int(size)
    return chrom_sizes

def collect_bigWig_paths(bgs_root: Path) -> list[Path]:
    """
    Return a list of paths of all bigWigs in `bgs_root` directory
    """
    if not bgs_root.is_dir():
        raise ValueError(f"{bgs_root} is not a directory")
    return list(bgs_root.rglob("*.bigWig"))

# TODO: argparse -> genomedata-style {--track <track_name> <track_file>}_i, i = 1 to n

class BigWigsBinner:
    def __init__(self, bigwig_paths: list[Path], chrom_sizes: dict[str, int], bin_size: int, parallel: bool = True):
        self.bin_size = bin_size
        self.parallel = parallel

        # for easy slicing out chromosomes
        # ranges IN BINNED TRACKS, not original genome
        chrom_starts_orig = np.cumsum([0] + list(chrom_sizes.values()))[:-1]
        # [If items(), keys(), values(), ... are called with
        # no intervening modifications to the dictionary,
        # the lists will directly correspond.](https://stackoverflow.com/a/835430)
        self.chrom_ranges = pd.DataFrame(
            "chrom_name": list(chrom_sizes.keys()),
            "start_bin": ceil(chrom_starts_orig / bin_size),
        )
        # TODO: how to deal with chromosomes that are divided between bins?

        # total length of genome divided evenly into bins of size `bin_size`
        self.n_bins = ceil(sum(chrom_sizes.values()) / bin_size)
        # will store the binned values for all bigWigs
        # in one giant tensor, a row for each bigWig
        self.binned_vals = torch.empty((len(bigwig_paths), self.n_bins), dtype=torch.float32)
    


def bin_chrom_bw(bigwig, chrom_sizes_entry: tuple, bin_size: int) -> list[float]:
    """
    Bin a single chromosome of `bigwig`, specified in `chrom_sizes_entry`
    as (name, size) of the chromosome to bin.
    Returns the mean signal values within each bin in a list, one element per bin.
    """
    n_bins = ceil(chrom_sizes_entry[1] / bin_size)
    print (f"Binning {chrom_sizes_entry[0]}")
    return bigwig.stats(chrom_sizes_entry[0], nBins=int(n_bins))

def bin_bigWig(bigwig, bin_size: int) -> dict[str, list[float]]:
    chrom_sizes = bigwig.chroms()
    # TEMPORARY for testing
    chrom_sizes = dict(itertools.islice(chrom_sizes.items(), 2))
    # For each chromosome in the bigWig, store
    # the list of bin (mean averaged) signal values
    # with the chromosome's name
    chrom_binned_series = {}
    for (name, size) in chrom_sizes.items():
        chrom_binned_series[name] = bin_chrom_bw(bigwig, (name,size), bin_size)
    return chrom_binned_series

def bin_bigWigs(bigwig_paths: list[Path], bin_size: int, parallel: bool) -> dict[dict[str, list[float]]]:
    """
    Bin all bigWigs into bins of size bin_size simultaneously,
    returns a dictionary of {name: [bin vals list]} with all chromosomes.
    """
    if parallel:
        n_threads = len(bigwig_paths)
    else:
        n_threads = 1
    print(f"Loading {len(bigwig_paths)} bigwigs in {n_threads} threads")

    bw_paths_strs = [str(path.absolute()) for path in bigwig_paths]
    bw_objs = None
    with ThreadPoolExecutor(max_workers=n_threads) as thr_pool:
        bw_objs = thr_pool.map(pyBigWig.open, bw_paths_strs)
    bw_objs = list(bw_objs)
    
    # Now that loaded, CPU-bound
    # Unfortunately, pyBigWig.bigWigFiles not picklable
    # with mp.Pool() as pool:
    #     bin_vals = pool.map(functools.partial(bin_bigWig, bin_size=bin_size), bw_objs)
    bin_vals = {}
    for i in len(bw_objs):
        bw_accession = bigwig_paths[i].stem
        bin_vals[bw_accession] = bin_bigWig(bw_objs[i], bin_size=bin_size)

    return bin_vals

'''
def binned_to_numpy(all_binned_vals_series: dict[dict[str, list[float]]], save_path: Path) -> None:
    """
    Save binned values to a numpy array containing
    every track but binned.
    Takes list with binned values for all tracks,
    each track contains all chromosomes as a
    dictionary of {chrom: [bin vals list]}
    """
    # First concatenate all binned values for
    # each track (all chromosomes) into one vector

def binned_to_DataFrame(all_binned_vals_series: list[dict[str, list[float]]], out_dir: Path) -> None:
    """
    Save binned values to a JSON and write to out_dir,
    structure values hierarchically:
        track name:
            chromosome
                bin vals list
            chromosome
                bin vals list
            ...,
        ...
    Takes list with binned values for all tracks,
    each track contains all chromosomes as a
    dictionary of {chrom: [bin vals list]}
    """
    with open(out_dir / "binned_vals.json", "w") as f:
        json_data = {
            "bin_size": BIN_SIZE,
            "tracks": all_binned_vals_series
        }
        json.dump(all_binned_vals_series, f, indent=4)
'''

if __name__ == "__main__":
    DATA_DIR = Path().resolve().parent.parent / "data"
    print(DATA_DIR.absolute())
    SIZES_FILE_PATH = DATA_DIR / "hg38.chrom.sizes"
    BIN_SIZE = 1000

    bw_paths = collect_bigWig_paths(DATA_DIR)
    print(f"Found {len(bw_paths)} bigWigs")

    bw_binned_tracks = bin_bigWigs(bw_paths, bin_size=1000, parallel=True)
    print(f"Loaded {len(bw_binned_tracks)} bedgraphs:")
    print(bw_binned_tracks[0].keys())
    print(bw_binned_tracks)