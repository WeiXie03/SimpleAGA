import numpy as np
import pandas as pd
from pyBedGraph import BedGraph
import multiprocessing as mp, functools
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import itertools
from math import ceil
from pathlib import Path
import argparse

def init_argparser(parser: argparse.ArgumentParser):
    parser.add_argument("data_dir", type=Path, help="Directory containing all bigWigs to be parsed, including within subdirectories.")
    parser.add_argument("bin_size", type=int, help="Size of bins to downsample the signal tracks to in base pairs. Also called resolution.")
    return parser

def collect_bedGraph_paths(bgs_root: Path):
    """
    Return a list of paths of all bedGraphs in `bgs_root` directory
    """
    if not bgs_root.is_dir():
        raise ValueError(f"{bgs_root} is not a directory")
    return list(bgs_root.rglob("*.bedGraph"))

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

if __name__ == "__main__":
    DATA_DIR = Path().resolve().parent.parent / "data"
    print(DATA_DIR.absolute())
    SIZES_FILE_PATH = DATA_DIR / "hg38.chrom.sizes"
    BIN_SIZE = 1000

    bg_paths = collect_bedGraph_paths(DATA_DIR)
    print(f"Found {len(bg_paths)} bedGraphs")
    chrom_sizes = parse_chromosome_sizes(SIZES_FILE_PATH)
    print("Parsed chromosome sizes:\n", chrom_sizes)
    chrom_sizes = dict(itertools.islice(chrom_sizes.items(), 5))

    bedgraph_objs = load_bedgraphs(bg_paths, chrom_sizes, SIZES_FILE_PATH, 1000, parallel=True)
    print(f"Loaded {len(bedgraph_objs)} bedgraphs")