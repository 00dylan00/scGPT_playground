"""Compute Embedding Distances.

Compute Euclidean Distance, Cosine Similarity, and Pearson Correlation between train embeddings and test embeddings.


Structure:
    1. Imports, Variables, Functions
    2. Load Data
    3. Compute Distances
    4. Save Results
"""

# 1. Imports, Variables, Functions
# imports
from scipy.spatial.distance import cosine
from scipy.spatial.distance import pdist, cdist
from typing import *
import pandas as pd, numpy as np, os, sys, h5py
import anndata as ad
import logging
from typing import *
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import json
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# variables
run_dir = os.path.join("..", "outputs", "run-24-10-12-01")
output_dir = os.path.join(run_dir, "outputs")

# functions

def load_run_output(input_dir: str) -> tuple:
    """Load the output of a run
    Args:
        input_dir (str): path to the run output directory
    Returns:
        loaded_variables (tuple): tuple of loaded variables
    """

    variables_to_load = [
        # "split",
        "predictions_test",
        "labels_test",
        "results_test",
        "all_outputs_test",
        "predictions_train",
        "labels_train",
        "results_train",
        "all_outputs_train",
        "adata_orig",
        "id2type",
    ]

    # initialize loaded variables as an empty tuple
    loaded_variables = ()

    # loop through variables
    for variable in variables_to_load:
        if variable.startswith("adata"):
            if False:
                # load everything
                loaded_variable = ad.read_h5ad(
                    os.path.join(input_dir, f"{variable}.h5ad")
                )

            else:
                # do not load everything
                loaded_variable = ad.read_h5ad(
                    os.path.join(input_dir, f"{variable}.h5ad"), backed="r"
                )

        else:
            with open(os.path.join(input_dir, f"{variable}.pkl"), "rb") as f:
                loaded_variable = pickle.load(f)

        # add the loaded variable to the tuple
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
    for i in range(len(output)):
        embeddings_i = output[i]["cell_emb"].numpy()

        if i == 0:
            embeddings = embeddings_i
        else:
            embeddings = np.concatenate((embeddings, embeddings_i), axis=0)
    return embeddings

# 2. Load Data
(
    # split,
    predictions_test,
    labels_test,
    results_test,
    all_outputs_test,
    predictions_train,
    labels_train,
    results_train,
    all_outputs_train,
    adata_orig,
    id2type,
) = load_run_output(run_dir)

# load json
with open(os.path.join(run_dir, "parameters.json"), "r") as f:
    parameters = json.load(f)

for k, v in parameters.items():
    print(f"{k}: {v}")

for i in range(0, len(all_outputs_test)):
    print(f"Fold {i}")
    scgpt_emb_test = merge_embeddings(all_outputs_test[i])
    logging.info(f"SCGPT Embeddings Test Shape: {scgpt_emb_test.shape}")

    scgpt_emb_train = merge_embeddings(all_outputs_train[i])
    logging.info(f"SCGPT Embeddings Train Shape: {scgpt_emb_train.shape}")
    break


# 3. Compute Distances
# Euclidean Distance
e_matrix_test = cdist(scgpt_emb_test, scgpt_emb_test, "euclidean")
e_matrix_train = cdist(scgpt_emb_train, scgpt_emb_train, "euclidean")

# Cosine Similarity
c_matrix_test = 1 - cdist(
    scgpt_emb_test, scgpt_emb_test, "cosine"
)  # 1 - cosine distance
c_matrix_train = 1 - cdist(scgpt_emb_train, scgpt_emb_train, "cosine")

# Pearson Correlation
p_matrix_test = 1 - cdist(
    scgpt_emb_test, scgpt_emb_test, "correlation"
) 
p_matrix_train = 1 - cdist(scgpt_emb_train, scgpt_emb_train, "correlation")

# 4. Save Results
# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save matrices as float16 numpy arrays
np.save(os.path.join(output_dir, "e_matrix_test.npy"), e_matrix_test.astype(np.float16))
np.save(os.path.join(output_dir, "e_matrix_train.npy"), e_matrix_train.astype(np.float16))

np.save(os.path.join(output_dir, "c_matrix_test.npy"), c_matrix_test.astype(np.float16))
np.save(os.path.join(output_dir, "c_matrix_train.npy"), c_matrix_train.astype(np.float16))

np.save(os.path.join(output_dir, "p_matrix_test.npy"), p_matrix_test.astype(np.float16))
np.save(os.path.join(output_dir, "p_matrix_train.npy"), p_matrix_train.astype(np.float16))

logging.info("All matrices saved as float16 in output directory.")


