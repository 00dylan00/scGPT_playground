"""
Compute Embedding Distances.

Compute Euclidean Distance, Cosine Similarity, and Pearson Correlation between train embeddings and test embeddings.

Structure:
    1. Imports, Variables, Functions
    2. Load Data
    3. Compute Distances
    4. Save Results
"""

# 1. Imports, Variables, Functions
# imports
from scipy.spatial.distance import cosine, cdist
from typing import *
import pandas as pd, numpy as np, os, sys, h5py
import anndata as ad
import logging
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import json
from concurrent.futures import ProcessPoolExecutor
import scanpy as sc

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# variables
run_dir = os.path.join("..", "outputs", "run-24-11-14-26")
output_dir = os.path.join(run_dir, "outputs")
data_type = "Raw GEx"

assert data_type in ["scGPT", "Raw GEx"], "Invalid data type. Choose from 'scGPT' or 'Raw GEx'."

# functions

def load_run_output(input_dir: str) -> tuple:
    """Load the output of a run
    Args:
        input_dir (str): path to the run output directory
    Returns:
        loaded_variables (tuple): tuple of loaded variables
    """
    variables_to_load = [
        "predictions_test",
        "labels_test",
        "results_test",
        "all_outputs_test",
        "predictions_valid",
        "labels_valid",
        "results_valid",
        "all_outputs_valid",
        "predictions_train",
        "labels_train",
        "results_train",
        "all_outputs_train",
        "adata_orig",
        "id2type",
        "train_indices",
        "valid_indices",
    ]

    global data_type

    loaded_variables = ()
    for variable in variables_to_load:
        if variable.startswith("adata"):
            
            if data_type == "Raw GEx":
                loaded_variable = ad.read_h5ad(os.path.join(input_dir, f"{variable}.h5ad"))
            else:

                loaded_variable = ad.read_h5ad(os.path.join(input_dir, f"{variable}.h5ad"), backed="r")
        else:
            with open(os.path.join(input_dir, f"{variable}.pkl"), "rb") as f:
                loaded_variable = pickle.load(f)
        loaded_variables += (loaded_variable,)

    print(f"Nº of loaded variables {len(loaded_variables)}")
    return loaded_variables


def merge_embeddings(output: List[Dict]) -> np.array:
    """Merge Embeddings
    Args:
        output (List[Dict]): List of dictionaries with embeddings
    Returns:
        np.array: Merged embeddings
    """
    embeddings = np.concatenate([output[i]["cell_emb"].numpy() for i in range(len(output))], axis=0)
    return embeddings


def compute_distance_chunk(embeddings_chunk, embeddings, metric):
    """Compute distances for a chunk of embeddings."""
    return cdist(embeddings_chunk, embeddings, metric)

def process_chunk(chunk, embeddings, metric):
    """Wrapper function for processing a chunk of embeddings."""
    return compute_distance_chunk(chunk, embeddings, metric)

def parallel_cdist(embeddings, metric, n_jobs=None):
    """Compute distance matrix in parallel using ProcessPoolExecutor."""
    # Determine chunk size (adjust as needed)
    chunk_size = embeddings.shape[0] // (n_jobs or os.cpu_count())
    chunks = [
        embeddings[i : i + chunk_size]
        for i in range(0, embeddings.shape[0], chunk_size)
    ]
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Pass each chunk directly to process_chunk instead of using lambda
        results = executor.map(process_chunk, chunks, [embeddings] * len(chunks), [metric] * len(chunks))
    
    # Concatenate the resulting chunks into a full matrix
    return np.vstack(list(results))

def minimal_normalization(adata: ad.AnnData) -> ad.AnnData:
    """Minimal Normalization
    Args:
        adata (ad.AnnData): AnnData object
    Returns:
        ad.AnnData: Normalized AnnData object
    """
    adata_copy = adata.copy()
    adata_copy.X = np.nan_to_num(adata_copy.X, nan=0)
    sc.pp.normalize_total(
    adata_copy, target_sum=1e6
    )
    sc.pp.log1p(adata_copy)
    adata_copy.X = np.nan_to_num(adata_copy.X, nan=0)
    return adata_copy

# 2. Load Data
(
    predictions_test,
    labels_test,
    results_test,
    all_outputs_test,
    predictions_valid,
    labels_valid,
    results_valid,
    all_outputs_valid,
    predictions_train,
    labels_train,
    results_train,
    all_outputs_train,
    adata_orig,
    id2type,
    train_indices,
    valid_indices
) = load_run_output(run_dir)

with open(os.path.join(run_dir, "parameters.json"), "r") as f:
    parameters = json.load(f)
for k, v in parameters.items():
    print(f"{k}: {v}")

for split_i in range(len(all_outputs_test)):

    if data_type == "Raw GEx":
        
        adata_test = adata_orig[adata_orig.obs[f"test_split_1"] == 1]
        adata_rest = adata_orig[adata_orig.obs[f"test_split_1"] == 0]

        adata_valid = adata_rest[valid_indices[0], :]
        adata_train = adata_rest[train_indices[0], :]
        
        adata_all = sc.concat([adata_test, adata_valid, adata_train], axis=0)    
    

        # minimal preprocessing
        adata_test = minimal_normalization(adata_test)
        adata_valid = minimal_normalization(adata_valid)
        adata_train = minimal_normalization(adata_train)
        adata_all = minimal_normalization(adata_all)

        emb_test = adata_test.X
        logging.info(f"Raw GEx Test Shape: {emb_test.shape}")

        emb_valid = adata_valid.X
        logging.info(f"Raw GEx Train Shape: {emb_valid.shape}")

        emb_train = adata_train.X
        logging.info(f"Raw GEx Train Shape: {emb_train.shape}")

        emb_all = adata_all.X
        logging.info(f"Raw GEx All Shape: {emb_all.shape}")
    
        name_extension = "raw_gex"


    elif data_type == "scGPT":

        # Only compute for the first fold as per original code structure
        emb_test = merge_embeddings(all_outputs_test[split_i])
        logging.info(f"SCGPT Embeddings Test Shape: {emb_test.shape}")

        emb_valid = merge_embeddings(all_outputs_valid[split_i])
        logging.info(f"SCGPT Embeddings Train Shape: {emb_valid.shape}")

        emb_train = merge_embeddings(all_outputs_train[split_i])
        logging.info(f"SCGPT Embeddings Train Shape: {emb_train.shape}")

        emb_all = np.concatenate([emb_test, emb_valid, emb_train], axis=0)
        logging.info(f"SCGPT Embeddings All Shape: {emb_all.shape}")

        name_extension = ""


    # 3. Compute & Save Distances
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Euclidean Distance
    e_matrix_test = parallel_cdist(emb_test, "euclidean")
    logging.info(f"Euclidean Distance Test Shape: {e_matrix_test.shape}")
    np.save(os.path.join(output_dir, f"{name_extension}e_matrix_test_parallel.split_{split_i+1}.npy"), e_matrix_test.astype(np.float16))
    del(e_matrix_test)

    e_matrix_valid = parallel_cdist(emb_valid, "euclidean")
    logging.info(f"Euclidean Distance valid Shape: {e_matrix_valid.shape}")
    np.save(os.path.join(output_dir, f"{name_extension}e_matrix_valid_parallel.split_{split_i+1}.npy"), e_matrix_valid.astype(np.float16))
    del(e_matrix_valid)

    e_matrix_train = parallel_cdist(emb_train, "euclidean")
    logging.info(f"Euclidean Distance Train Shape: {e_matrix_train.shape}")
    np.save(os.path.join(output_dir, f"{name_extension}e_matrix_train_parallel.split_{split_i+1}.npy"), e_matrix_train.astype(np.float16))
    del(e_matrix_train)

    e_matrix_all = parallel_cdist(emb_all, "euclidean")
    logging.info(f"Euclidean Distance All Shape: {e_matrix_all.shape}")
    np.save(os.path.join(output_dir, f"{name_extension}e_matrix_all_parallel.split_{split_i+1}.npy"), e_matrix_all.astype(np.float16))
    del(e_matrix_all)

    # Cosine Similarity
    c_matrix_test = 1 - parallel_cdist(emb_test, "cosine")
    logging.info(f"Cosine Similarity Test Shape: {c_matrix_test.shape}")
    np.save(os.path.join(output_dir, f"{name_extension}c_matrix_test_parallel.split_{split_i+1}.npy"), c_matrix_test.astype(np.float16))
    del(c_matrix_test)

    c_matrix_valid = 1 - parallel_cdist(emb_valid, "cosine")
    logging.info(f"Cosine Similarity Train Shape: {c_matrix_valid.shape}")
    np.save(os.path.join(output_dir, f"{name_extension}c_matrix_valid_parallel.split_{split_i+1}.npy"), c_matrix_valid.astype(np.float16))
    del(c_matrix_valid)

    c_matrix_train = 1 - parallel_cdist(emb_train, "cosine")
    logging.info(f"Cosine Similarity Train Shape: {c_matrix_train.shape}")
    np.save(os.path.join(output_dir, f"{name_extension}c_matrix_train_parallel.split_{split_i+1}.npy"), c_matrix_train.astype(np.float16))
    del(c_matrix_train)

    c_matrix_all = 1 - parallel_cdist(emb_all, "cosine")
    logging.info(f"Cosine Similarity All Shape: {c_matrix_all.shape}")
    np.save(os.path.join(output_dir, f"{name_extension}c_matrix_all_parallel.split_{split_i+1}.npy"), c_matrix_all.astype(np.float16))

    # Pearson Correlation
    p_matrix_test = 1 - parallel_cdist(emb_test, "correlation")
    logging.info(f"Pearson Correlation Test Shape: {p_matrix_test.shape}")
    np.save(os.path.join(output_dir, f"{name_extension}p_matrix_test_parallel.split_{split_i+1}.npy"), p_matrix_test.astype(np.float16))
    del(p_matrix_test)

    p_matrix_valid = 1 - parallel_cdist(emb_valid, "correlation")
    logging.info(f"Pearson Correlation Train Shape: {p_matrix_valid.shape}")
    np.save(os.path.join(output_dir, f"{name_extension}p_matrix_valid_parallel.split_{split_i+1}.npy"), p_matrix_valid.astype(np.float16))
    del(p_matrix_valid)


    p_matrix_train = 1 - parallel_cdist(emb_train, "correlation")
    logging.info(f"Pearson Correlation Train Shape: {p_matrix_train.shape}")
    np.save(os.path.join(output_dir, f"{name_extension}p_matrix_train_parallel.split_{split_i+1}.npy"), p_matrix_train.astype(np.float16))
    del(p_matrix_train)

    p_matrix_all = 1 - parallel_cdist(emb_all, "correlation")
    logging.info(f"Pearson Correlation All Shape: {p_matrix_all.shape}")
    np.save(os.path.join(output_dir, f"{name_extension}p_matrix_all_parallel.split_{split_i+1}.npy"), p_matrix_all.astype(np.float16))

    logging.info("All matrices saved as float16 in output directory.")
