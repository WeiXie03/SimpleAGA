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
from pomegranate.distributions import Normal, Categorical
from pomegranate.hmm import DenseHMM
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
    # print(chrom_arr)
    nan_cols = torch.any(torch.isnan(chrom_arr), 0)
    # print("NaN columns:")
    # print(nan_cols)
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

def train(model, observations: torch.Tensor, n_procs: int) -> None:
    """
    Train the model on the given data
    n_procs: number of processes to use for parallelization
    all_binneds: a dictionary of chromosome names to 2D tensors, each row a bigWig track
    """
    observations, missing_idx = omit_missing(observations)
    # TODO: keep track of the indices of the omitted positions
    # self.missing_rngs_tbl = pd.DataFrame()

    #print([obs for obs, _ in obs_omitidx_pairs_list])
    #observations = torch.vstack([obs for obs, _ in obs_omitidx_pairs_list])

    # input: (n_tracks, n_bins), emissions convention: (n_labels, n_tracks)
    observations = observations.transpose(0, 1).unsqueeze(0)
    print("Processesed observations after omitting missing values:", observations.shape)
    print(observations)

    model.fit(observations)

def run_chrom(num_labels: int, n_procs: int, observations: torch.Tensor, tol=10, max_iter=None, handle_missing=HandleMissingStrategy.OMIT) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Full training pipeline--initialize, train and return the model parameters
    Model parameters--2 2D tensors and 1 3D tensor:
        - Emission probabilities--Normals: means and variances, both (n_labels, n_tracks)
        - Transition probabilities--Categorical: (n_labels, n_labels)

    num_labels: number of labels
    n_procs: number of processes to use for parallelization in omitting missing values
    all_binneds: a dictionary of chromosome names to 2D tensors, each row a bigWig track
    handle_missing: how to handle missing values
    """
    # Random initialize the model
    observations, missing_idx = omit_missing(observations)
    # one dim per track, init covs as identity
    init_means = observations.mean(1)
    dists = [Normal(init_means, torch.eye(len(init_means))) for state in range(num_labels)]
    print("Initialized emission distributions:", dists)
    for d in dists:
        print('means:', d.means, '\ncovs:', d.covs)
    # uniform prob
    init_transitions = torch.ones(num_labels, num_labels) / num_labels
    init_starts = torch.ones(num_labels) / num_labels
    init_ends = torch.ones(num_labels) / num_labels

    # Normals will *implicitly* be cast to n_tracks dimensional _through_ `fit(observations)`
    if max_iter == None and tol > 0:
        # model = DenseHMM([Normal()] * num_labels, tol=tol, verbose=True)
        model = DenseHMM(dists, edges=init_transitions, starts=init_starts, ends=init_ends, tol=tol, verbose=True)
    elif tol == None and max_iter > 0:
        # model = DenseHMM([Normal()] * num_labels, max_iter=max_iter, verbose=True)
        model = DenseHMM(dists, edges=init_transitions, starts=init_starts, ends=init_ends, max_iter=max_iter, verbose=True)
    else:
        print("Error: either tol or max_iter must be positive")
        model = DenseHMM(dists, edges=init_transitions, starts=init_starts, ends=init_ends, verbose=True)
    print("Initialized model:", model)
    train(model, observations, n_procs)
    
    # Return the model parameters
    emit_means = torch.vstack([d.means for d in model.distributions])
    # will be 3D tensor (n_labels, n_tracks, n_tracks)?
    emit_covs = torch.stack([d.covs for d in model.distributions])
    transitions = model.edges
    return emit_means, emit_covs, transitions

def main(num_labels: int, n_procs: int, all_binneds: dict[str, torch.Tensor], tol=10, max_iter=None):
    for chrom, tensor in all_binneds.items():
        print(chrom, ':', tensor.shape)
        # Train the model
        emit_means, emit_covs, transitions = run_chrom(num_labels, n_procs, tensor, tol=tol, max_iter=max_iter)
        print("Emission means:")
        print(emit_means)
        # print("Emission covariances:")
        # print(emit_covs)
        print("Transitions:")
        print(transitions)
    
    return emit_means, emit_covs, transitions

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    N_PROCS = os.cpu_count()

    # ================== #
    # Test 1: Dummy data #
    # ================== #
    tensor = torch.randn((4, 16))
    # print(tensor)
    emit_means, emit_covs, transitions = run_chrom(2, N_PROCS, tensor, max_iter=100)

    # # =================== #
    # # Test 2: Toy chroms #
    # # =================== #
    # # DATA_DIR = Path("../../data")
    # # chrom_sizes_path = DATA_DIR / "hg38.chrom.sizes"
    # DATA_DIR = Path("../tests/test_data")
    # chrom_sizes_path = DATA_DIR / "toy.chrom.sizes"

    # chrom_sizes = _util.parse_chromosome_sizes(chrom_sizes_path)
    # chrom_sizes = pd.DataFrame(chrom_sizes.items(), columns=["name", "size"])
    # # Load in the binned data (tensors)
    # all_binneds = load_all_chrom_tensors(chrom_sizes["name"], DATA_DIR/"combined_out")

    # # Train the model
    # emit_means, emit_covs, transitions = main(2, N_PROCS, all_binneds, max_iter=100)

    # =================== #
    # Test 3: CD14 positive monocyte subset
    #   - H3K27ac,
    #   - H3K27me3
    # =================== #
    ROOT_DIR = Path("/home/tom/Public/projSAGA/data/CD14-sub")
    DATA_DIR = ROOT_DIR / "rep1" / "binned21"
    chrom_sizes_path = ROOT_DIR / "hg38_chr21.chrom.sizes"

    chrom_sizes = _util.parse_chromosome_sizes(chrom_sizes_path)
    chrom_sizes = pd.DataFrame(chrom_sizes.items(), columns=["name", "size"])
    # Load in the binned data (tensors)
    all_binneds = load_all_chrom_tensors(chrom_sizes["name"], DATA_DIR)

    # Train the model
    emit_means, emit_covs, transitions = main(25, N_PROCS, all_binneds, max_iter=1000)