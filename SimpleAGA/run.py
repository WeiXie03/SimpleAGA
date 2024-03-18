"""
Command line interface to run the SimpleAGA model on a set of chromosomes.

Arguments
---
`--rep_dir`: The directory containing the tensor files for the chromosomes to analyze.
`--chrom-sizes`: The path to the TSV file containing the sizes of the chromosomes to analyze.
`--out-dir`: The directory to save the output files to.
`--res`: The resolution (i.e. bin size) of the data in the tensor files.
`--minibatch-chunk-size`: The size of each randomly sampled contiguous chunk of the genome to analyze in each minibatch.

Optional:
`--n-labels`: The number of labels to use for the model. Default is 16.
`--n-iter`: The number of iterations to use for the model. Default is 1000.
`--minibatch-frac`: The fraction of the genome to analyze in each minibatch. Default is 0.3.
TODO:
`--model`: The path to the trained model to use for prediction.
`--coords-bed`: The path to the BED file containing the coordinates of the bins in the tensor files.
"""

from __future__ import annotations
from model import *
import os, sys
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import argparse
#import wandb

def init_parser():
    parser = argparse.ArgumentParser(description="Run the SimpleAGA model on a set of chromosomes.")
    parser.add_argument("--rep-dir", type=Path, required=True, help="The directory containing the tensor files for the chromosomes to analyze.")
    parser.add_argument("--chrom-sizes", type=Path, required=True, help="The path to the TSV file containing the sizes of the chromosomes to analyze.")
    parser.add_argument("--out-dir", type=Path, required=True, help="The directory to save the output files to.")
    parser.add_argument("--res", type=int, required=True, help="The resolution (i.e. bin size) of the data in the tensor files.")
    parser.add_argument("--minibatch-chunk-size", type=int, required=True, help="The size of each randomly sampled contiguous chunk of the genome to analyze in each minibatch.")
    parser.add_argument("--n-labels", type=int, default=16, help="The number of labels to use for the model. Default is 16.")
    parser.add_argument("--n-iter", type=int, default=1000, help="The number of iterations to use for the model. Default is 1000.")
    parser.add_argument("--minibatch-frac", type=float, default=0.3, help="The fraction of the genome to analyze in each minibatch. Default is 0.3.")
    return parser

def collect_chroms_info(rep_dir: Path, chrom_sizes_path: Path) -> pd.DataFrame:
    """
    Collects the name and size of each chromosome specified in the chrom_sizes file, as well as
    the path to the corresponding tensor file in `rep_dir`.
    Returns the information as a DataFrame with columns "size" and "path", and names as the index.
    
    Args
    ---
    `chrom_sizes_path`: A tab-separated file with first two columns choromsome name and size.

    `rep_dir`: Must include all chromosomes want to analyze, one tensor each. Columns should be assays, rows should be bins.
    """
    if not rep_dir.is_dir():
        raise ValueError("Invalid path for directory of chromsome tensors, directory not found.")
    if not chrom_sizes_path.is_file():
        raise ValueError("Invalid path for chromosome sizes TSV file, file not found.")

    # read the chromosome names into the index
    chroms_meta_tbl = pd.read_csv(chrom_sizes_path, sep='\t', index_col=0, header=None, names=["size"])

    tensor_paths = []
    for path in rep_dir.iterdir():
        if path.suffix == ".pt":
            tensor_paths.append(str(path))
    # directly generate series now so can generate index from the file names in parallel
    tensor_paths = pd.Series(tensor_paths)
    tensor_paths.index = tensor_paths.apply(lambda x: str(Path(x).stem))
    
    chroms_meta_tbl["path"] = tensor_paths
    # don't take the tensors for chromsomes not in the chrom_sizes file
    chroms_meta_tbl.dropna(inplace=True, subset=["size", "path"])

    return chroms_meta_tbl

if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()

    # collect the chromosome information
    chroms_meta_tbl = collect_chroms_info(args.rep_dir, args.chrom_sizes)

    rand_gen = np.random.default_rng()
    # initialize the model
    mngr = RunManager(chroms_meta_tbl["size"], args.res, num_labels=args.n_labels, n_iter=args.n_iter, rand_gen=rand_gen)

    mngr.run_batch(chroms_meta_tbl["path"], args.out_dir, args.res, args.minibatch_frac, args.minibatch_chunk_size)

    print("Model run complete.")
    sys.exit(0)