import numpy as np
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

def collect_bedGraph_paths(bgs_root: Path):
    """
    Return a list of paths of all bedGraphs in `bgs_root` directory
    """
    if not bgs_root.is_dir():
        raise ValueError(f"{bgs_root} is not a directory")
    return list(bgs_root.rglob("*.bedGraph"))

def collect_bigWig_paths(bgs_root: Path):
    """
    Return a list of paths of all bigWigs in `bgs_root` directory
    """
    if not bgs_root.is_dir():
        raise ValueError(f"{bgs_root} is not a directory")
    return list(bgs_root.rglob("*.bigWig"))

def parse_chromosome_sizes(chrom_sizes_file: Path):
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

# TODO: argparse -> genomedata-style {--track <track_name> <track_file>}_i, i = 1 to n

def load_bedgraph(bedgraph_file: Path, chrom_name: str, chrom_sizes_path: Path, bin_size: int) -> BedGraph:
    """
    Load bedgraph into memory
    """
    chrom_names = [chrom_name]
    bedgraph = BedGraph(chrom_sizes_path, bedgraph_file, chrom_names)
    print("Loading", bedgraph_file, "for", chrom_name)
    bedgraph.load_chrom_bins(chrom_names, bin_size)
    print("Loaded", bedgraph_file, "for", chrom_name)
    return bedgraph

def load_bedgraphs(bedgraph_paths: list[Path], chrom_sizes: dict, chrom_sizes_path: Path, bin_size: int, parallel: bool) -> list[BedGraph]:
    """
    Load all bedgraphs into memory
    """
    # Every signal track covers all chromosomes, load all in parallel
    if parallel:
        n_threads = len(chrom_sizes) * len(bedgraph_paths)
    else:
        n_threads = 1
    print(f"Loading {len(chrom_sizes)} chromosomes for each of {len(bedgraph_paths)} bedgraphs in {n_threads} threads")

    bedgraphs = []
    with ThreadPoolExecutor(max_workers=n_threads) as thr_pool:
        # TODO: probably getting a deadlock right now, bedgraph file can only be accessed by one thread at a time?
        #   => just load each chrom of bedgraph sequentially
        for chrom_name in chrom_sizes.keys():
            for bedgraph_path in bedgraph_paths:
                bedgraphs.append(thr_pool.submit(load_bedgraph, bedgraph_path, chrom_name, chrom_sizes_path, bin_size))
    
    return bedgraphs

def bin_bedgraphs(bedgraphs: list[BedGraph], chrom_sizes: dict, parallel: bool):
    """
    Bin all bedgraphs into bins of size bin_size simultaneously
    """
    pass
    # First get all binned values into one list with all bedgraphs(tracks)
    # Now that loaded, CPU-bound
    # binned_vals_lists = []

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
    chrom_sizes = dict(itertools.islice(chrom_sizes.items(), 5))
    # For each chromosome in the bigWig, store
    # the list of bin (mean averaged) signal values
    # with the chromosome's name
    chrom_binned_series = {}
    for (name, size) in chrom_sizes.items():
        chrom_binned_series[name] = bin_chrom_bw(bigwig, (name,size), bin_size)
    return chrom_binned_series

def bin_bigWigs(bigwig_paths: list[Path], bin_size: int, parallel: bool) -> list[dict[str, list[float]]]:
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
    bin_vals = [bin_bigWig(bw_obj, bin_size=bin_size) for bw_obj in bw_objs]

    return bin_vals

def init_argparser(parser: argparse.ArgumentParser):
    parser.add_argument("data_dir", type=Path, help="Directory containing all bigWigs to be parsed, including within subdirectories.")
    parser.add_argument("bin_size", type=int, help="Size of bins to downsample the signal tracks to in base pairs. Also called resolution.")
    return parser

if __name__ == "__main__":
    DATA_DIR = Path().resolve().parent.parent / "data"
    print(DATA_DIR.absolute())
    SIZES_FILE_PATH = DATA_DIR / "hg38.chrom.sizes"
    BIN_SIZE = 1000

    """========= bedGraphs ========"""
    # bg_paths = collect_bedGraph_paths(DATA_DIR)
    # print(f"Found {len(bg_paths)} bedGraphs")
    # chrom_sizes = parse_chromosome_sizes(SIZES_FILE_PATH)
    # print("Parsed chromosome sizes:\n", chrom_sizes)
    # chrom_sizes = dict(itertools.islice(chrom_sizes.items(), 5))

    # bedgraph_objs = load_bedgraphs(bg_paths, chrom_sizes, SIZES_FILE_PATH, 1000, parallel=True)
    # print(f"Loaded {len(bedgraph_objs)} bedgraphs")

    """========= bigWigs ========"""
    bw_paths = collect_bigWig_paths(DATA_DIR)
    print(f"Found {len(bw_paths)} bigWigs")

    bw_binned_tracks = bin_bigWigs(bw_paths, bin_size=1000, parallel=True)
    print(f"Loaded {len(bw_binned_tracks)} bedgraphs")