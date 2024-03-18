'''
The SAGA, a Hidden Markov Model, and associated
code for training and saving the model.
'''

from math import ceil
import numpy as np
import torch
import pandas as pd
from pyBedGraph import BedGraph
import multiprocessing as mp
import functools
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import itertools
from enum import Enum
import os
from pathlib import Path
import pickle
from hmmlearn import hmm
from SimpleAGA import _util

def load_chrom_tensor_to_array(path: Path) -> np.ndarray:
    """
    Load binned data for chromosome chr_name from its torch .pt file in data_dir,
    assuming each .pt file is named <chr_name>.pt
    Returns a tuple of the chromosome name and the loaded tensor
    """
    print(f"Loading {path}...")
    try:
        data = torch.load(path).numpy()
    except FileNotFoundError:
        print(f"{path} not found")
    finally:
        return data

def load_chrom_tensor_to_array_spec_coords(intervs_info_df_chr: "pd.DataFrame | pd.core.groupby.generic.DataFrameGroupBy", binned_chroms_paths: pd.Series) -> np.ndarray:
    """
    Same as load_chrom_tensor_to_array, but takes a DataFrame of
    genomic coordinates for the chromosome:
    DataFrame must be the first 3 columns of a BED file:
        - Chromosome name
        - Start position
        - End position
    """
    if not isinstance(intervs_info_df_chr, pd.DataFrame):
        raise TypeError("intervs_info_df_chr must be a DataFrame")

    else:
        # Path of data for this chromosome
        data = load_chrom_tensor_to_array(binned_chroms_paths[intervs_info_df_chr.index[0]])
        # Subset to coordinates
        # intervs_info_df_chr = chrom, *start*, *end*
        if len(intervs_info_df_chr.index) > 0:
            return data[intervs_info_df_chr.iloc[:,1]:intervs_info_df_chr.iloc[:,2],:]
        else:
            # No coordinates
            return np.empty(data.shape)

def omit_missing(chrom_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Omit positions at which _any_ of the tracks are missing values from all_binneds
    Returns:
        - a new tensor with the missing positions omitted, and
        - a tensor of the indices of the omitted positions
    For a single chromosome (represented by one 2D tensor, each row a bigWig track)
    """
    # Find columns with NaN values
    # "Any" _through_ columns, reducing the 1th dimension
    nan_rows = np.any(np.isnan(chrom_arr), 1)
    # Convert from bool mask to indices
    nan_rows_idx = np.where(nan_rows)[0]
    # Remove rows with NaN values
    return chrom_arr[~nan_rows,:], nan_rows_idx

class RunManager():
    """
    Manages loading of training data, training and running model,
    parsing and saving the results.
    """
    def __init__(self, chrom_sizes: pd.Series, resolution: int, coords: pd.DataFrame = None,
                 model: hmm.GaussianHMM = None,
                 num_labels: int = None, n_iter: int = None,
                 rand_gen: np.random.BitGenerator = np.random.default_rng()):
        """
        Optionally initializes a model.

        Args
        ---
        chrom_sizes: A Series of the sizes of the chromosomes in
        the data to train on. Must have:
            - An index of the chromosome names.
            - A column "size": The sizes of the chromosome.
        
        resolution: The resolution (bin size) in bp the data were binned with.

        num_labels: The number of labels (hidden states) for the model.

        coords: Optionally specify genomic coordinates to use (subset from the data).
        DataFrame must be the first 3 columns of a BED file:
            - Chromosome name
            - Start position
            - End position
        """
        self.chrom_sizes = chrom_sizes
        self.bin_size = resolution
        self.rand_gen = rand_gen

        if isinstance(coords, pd.DataFrame):
            self.coords = coords
        else:
            self.coords = None

        if isinstance(model, hmm.GaussianHMM):
            self.model = model
        elif num_labels != None and n_iter != None:
            self.model = hmm.GaussianHMM(n_components=num_labels, covariance_type="full", n_iter=n_iter, verbose=True)
        else:
            raise ValueError("Must provide either a model, or num_labels and n_iter.")
        
    def load_all_chrom_arrays(self, binned_chroms_paths: pd.Series,
                              n_threads=None) -> list[np.ndarray]:
        """
        Maps load_chrom_tensor_to_array to all chromosomes in binned_chroms_paths,
        returning them all in a list in the same order as in binned_chroms_paths.

        Takes a DataFrame including the paths to the binned data for each chromosome.
        Index must be the chromosome names. Must include a column "path" of paths.
        Must be PyTorch tensors; extension ".pt".
        """
        if not isinstance(n_threads, int):
            if n_threads != None:
                print("Error: n_threads must be an int")
            n_threads = len(binned_chroms_paths.index)
        
        print(f"Loading {len(binned_chroms_paths.index)} chromosomes with {n_threads} threads...")

        if isinstance(self.coords, pd.DataFrame):
            chrom_arrs = self.coords.parallel_apply(load_chrom_tensor_to_array, axis=1,
                                               binned_chroms_paths=binned_chroms_paths)
        elif self.coords == None:
            with ThreadPoolExecutor(n_threads) as pool:
                chrom_arrs = pool.map(load_chrom_tensor_to_array, binned_chroms_paths.tolist())
        
        return chrom_arrs
    
    def train_batch(self, train_seqs: list[np.ndarray], seq_lens: list[int],
                    frac: float = 0.03, subsample_len: int = None,
                    n_procs: int = None):
        """
        Trains the model on random subsequences totalling `frac` of the sum of the lengths
        of the given sequences.
        If `subsample_len` is given, samples `n` random subsequences each of length
        `subsample_len`, where `n`*`subsample_len` = `frac` * $sum_{seq} |seq|$.
        Otherwise, samples one random subsequence from each given sequence of length
        `frac`*length(seq).
        """
        if not isinstance(n_procs, int):
            if n_procs != None:
                print("Error: n_procs must be an int")
            n_procs = min(mp.cpu_count(), len(train_seqs))
        
        if not isinstance(frac, float):
            print("Error: frac must be a float")

        # print("Training sequence data shapes:")
        # print([seq.shape for seq in train_seqs])
        
        if len(train_seqs) <= 0:
            raise ValueError("Must provide at least one training sequence.")
        elif len(train_seqs) == 1:
            mini_start, mini_end = _util.slice_rand_subseq_idx(train_seqs[0], frac)
            self.model.fit(train_seqs[0][mini_start:mini_end])
        else:
            # one from each sequence
            if subsample_len == None: 
                subseqs = _util.sample_minibatches(train_seqs, frac=frac, lens=seq_lens, rand_gen=self.rand_gen)
            # n_samples random subsequences
            else:
                subseqs = _util.sample_minibatches(train_seqs, frac=frac, lens=seq_lens, subseq_len=subsample_len, rand_gen=self.rand_gen)

            self.model.fit(np.vstack(np.concatenate(subseqs, axis=0)), lengths=_util.mp_arrays_lens(subseqs, n_procs))

    def gen_posteriors_tbl(self, train_seqs: list[np.ndarray], seq_lens: list[int], missing_idxs_list: list[np.ndarray], n_procs: int = None
                           ) -> pd.DataFrame:
        if not isinstance(n_procs, int):
            if n_procs != None:
                print("Error: n_procs must be an int, setting to min(n_procs, len(train_seqs))")
            n_procs = min(mp.cpu_count(), len(train_seqs))
        
        with mp.Pool(n_procs) as pool:
            fixed_gen = functools.partial(_util.gen_chrom_intervs_tbl, chrom_sizes=self.chrom_sizes, bin_size=self.bin_size)
            intervs_tbls = pool.starmap(fixed_gen, list(zip(self.chrom_sizes.index, train_seqs, missing_idxs_list)))
        
        combined_intervs_tbl = pd.concat(list(intervs_tbls), axis=0, ignore_index=True)

        if len(train_seqs) == 1:
            posteriors = self.model.predict_proba(train_seqs[0])
        else:
            posteriors = self.model.predict_proba(train_seqs, lengths=seq_lens)
        
        tbl_posterios_half = pd.DataFrame(posteriors, columns=[f"posterior{i}" for i in range(1, self.model.n_components + 1)])

        return pd.concat([combined_intervs_tbl, tbl_posterios_half], axis=1)
        
    def display_results(self, posteriors_tbl: pd.DataFrame):
        """
        Displays the results of the model.
        """
        print("Training converged:", self.model.monitor_.converged)
        print(self.model.monitor_)

        print("Parsed posteriors table: first five rows")
        print(posteriors_tbl.head())
        
    def run_batch(self, binned_chroms_paths: pd.Series, save_dir: Path = None,
                  train: bool = True, minibatch_frac: float = None, subsample_len: int = None,
                  n_procs: int = None):
        """
        Full pipeline for a batch of data:
        1. Load data.
        2. Run model.
        3. Parse results, optionally save.

        Args
        ----
        binned_chroms_paths: A DataFrame including the paths to the binned data for each chromosome.
        Index must be the chromosome names, must include a column "path" of paths.
        Must be PyTorch tensors; extension ".pt".

        """
        if not isinstance(n_procs, int):
            if n_procs != None:
                print("Error: n_procs must be an int")
            n_procs = min(mp.cpu_count(), len(binned_chroms_paths.index))

        binned_tensors = self.load_all_chrom_arrays(binned_chroms_paths)

        with ProcessPoolExecutor(n_procs) as pool:
            pruned_missidx_pairs_list = pool.map(omit_missing, binned_tensors)
        pruned_tensors, chroms_miss_idx = zip(*pruned_missidx_pairs_list)
        print(f"Processed {len(pruned_tensors)} chromosomes")
        # print("Missing indices:")
        # for (chrom, miss_idx) in zip(self.chrom_sizes.index, chroms_miss_idx):
        #     print(f"{chrom}: {miss_idx}")

        # Mark each processed chromosome tensor length with its chromosome name
        procd_chroms_lens = pd.Series(_util.mp_arrays_lens(pruned_tensors, n_procs), index=self.chrom_sizes.index)
        # Run model
        if train:
            print("Training HMM...")
            self.train_batch(pruned_tensors, procd_chroms_lens.to_numpy(), frac=minibatch_frac, subsample_len=subsample_len, n_procs=n_procs)

        print("Generating posteriors table...")
        posteriors_tbl = self.gen_posteriors_tbl(pruned_tensors, procd_chroms_lens, chroms_miss_idx, n_procs)

        self.display_results(posteriors_tbl)

        if isinstance(save_dir, Path):
            if not save_dir.exists():
                os.makedirs(save_dir)

            np.save(save_dir/"emissions_means.npy", self.model.means_)
            np.save(save_dir/"emissions_covars.npy", self.model.covars_)
            np.save(save_dir/"transitions.npy", self.model.transmat_)
            with open(save_dir/"hmmlearn_model.pkl", "wb") as f:
                pickle.dump(self.model, f)

            posteriors_tbl.to_csv(save_dir/"parsed_posteriors.csv")

# TODO: Add command line interface
def main(chrom_sizes_path: Path, resolution: int,
        coords_path: Path = None, model_path: Path = None,
         num_labels: int = None, n_iter: int = None):
    pass
    # if isinstance(coords_path, Path):
    #     # BED file: first 3 cols = chrom, start, end
    #     coords = pd.read_csv(coords_path, sep="\t", header=None, index_col=0, usecols=[0,1,2])