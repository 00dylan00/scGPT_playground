"""Load Data
Structure:
    1. Imports, Variables, Functions
    2. Load Data
"""

# 1. Imports, Variables, Functions
# imports
import pandas as pd, numpy as np, os, sys
import anndata as ad
import logging
from typing import *
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.metrics import confusion_matrix, classification_report
import sys
sys.path.append(os.path.join("..", ".."))
from src.utils import utils as ut
from src.utils import viz as vz
from src.utils import io 
logging.basicConfig(level=logging.INFO)

# variables
run_dir = os.path.join("/aloy/home/ddalton/projects/scGPT_playground","outputs","run-25-09-13-21") # A 
output_dir = os.path.join(run_dir, "outputs")


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# functions

# 2. Load Data
(
    # split,
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
    valid_indices,
) = io.load_run_output(run_dir)

# load json

with open(os.path.join(run_dir, "parameters.json"), "r") as f:
    parameters = json.load(f)

for k, v in parameters.items():
    print(f"{k}: {v}")


split_idx = 0

# load all adata
adata_test =  ad.read_h5ad(
    os.path.join(run_dir, f"adata_test_{split_idx+1}.h5ad"), backed="r"
)
adata_test.obs.reset_index(drop=True, inplace=True)  
adata_valid =  ad.read_h5ad(
    os.path.join(run_dir, f"adata_valid_{split_idx+1}.h5ad"), backed="r"
)
adata_valid.obs.reset_index(drop=True, inplace=True)  
adata_train =  ad.read_h5ad(
    os.path.join(run_dir, f"adata_train_{split_idx+1}.h5ad"), backed="r"
)
adata_train.obs.reset_index(drop=True, inplace=True)  
"""Compute similarity matrixes
"""

# 1. Imports, Variables, Functions
# imports
from typing import *
from scipy.spatial.distance import cdist
import itertools
import obonet
import networkx as nx
import sys
sys.path.append(os.path.join("..", ".."))
from src.utils import utils as ut
from src.utils import viz as vz
import logging
import os
# variable
split_idx = 0

# functions


# 2. Load Data
# load all adata
adata_test =  ad.read_h5ad(
    os.path.join(run_dir, f"adata_test_{split_idx+1}.h5ad"), backed="r"
)
adata_test.obs.reset_index(drop=True, inplace=True)  
adata_valid =  ad.read_h5ad(
    os.path.join(run_dir, f"adata_valid_{split_idx+1}.h5ad"), backed="r"
)
adata_valid.obs.reset_index(drop=True, inplace=True)  
adata_train =  ad.read_h5ad(
    os.path.join(run_dir, f"adata_train_{split_idx+1}.h5ad"), backed="r"
)
adata_train.obs.reset_index(drop=True, inplace=True)  


X_train = adata_train.X
X_train = np.where(np.isnan(X_train), 0, X_train)
X_valid = adata_valid.X
X_valid = np.where(np.isnan(X_valid), 0, X_valid)
X_test = adata_test.X
X_test = np.where(np.isnan(X_test), 0, X_test)

# compute correlation similarity between all embeddings
c_matrix_train =1 -  cdist(X_train, X_train, "correlation")
c_matrix_train.astype(np.float16).tofile(os.path.join(output_dir, "c_matrix_train.bin"))
del c_matrix_train
print("train done")

c_matrix_valid =1 -  cdist(X_valid, X_valid, "correlation")
c_matrix_valid.astype(np.float16).tofile(os.path.join(output_dir, "c_matrix_valid.bin"))
del c_matrix_valid
print("valid done")

c_matrix_test =1 -  cdist(X_test, X_test, "correlation")
c_matrix_test.astype(np.float16).tofile(os.path.join(output_dir, "c_matrix_test.bin"))
del c_matrix_test
print("test done")

# compute correlation similarity between sets
c_matrix_test_vs_train =1 -  cdist(X_test, X_train, "correlation")
c_matrix_test_vs_train.astype(np.float16).tofile(os.path.join(output_dir, "c_matrix_test_vs_train.bin"))
del c_matrix_test_vs_train
print("test vs train done")

c_matrix_valid_vs_train =1 - cdist(X_valid, X_train, "correlation")
c_matrix_valid_vs_train.astype(np.float16).tofile(os.path.join(output_dir, "c_matrix_valid_vs_train.bin"))
del c_matrix_valid_vs_train
print("valid vs train done")