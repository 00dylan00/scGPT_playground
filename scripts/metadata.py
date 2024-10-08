import numpy as np, os, sys, pandas as pd, scanpy as sc
import anndata as ad
import logging
from tqdm import tqdm
from typing import *
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
from matplotlib import pyplot as plt
from datetime import datetime
import pickle


folder_run_name = "pp_data-24-09-02-01"

data_folder = os.path.join("/aloy/home/ddalton/projects/scGPT_playground/data/",
                            folder_run_name)
data_path = os.path.join(data_folder, "data.h5ad")

adata = sc.read(data_path)

# save metadata

metadata_txt = "All Human Diseases"

# compute metadata values
n_genes = adata.X.shape[1]
n_gex = adata.X.shape[0]    
n_non_nan_genes = np.sum(~np.isnan(adata.X), axis=0)
n_non_nan_gex = np.sum(~np.isnan(adata.X), axis=1)
genes_std = np.nanstd(adata.X, axis=0)
gex_std = np.nanstd(adata.X, axis=1)

metadata = {"metadata": metadata_txt,
            "n_genes": n_genes,
            "n_gex": n_gex,
            "n_non_nan_genes": n_non_nan_genes,
            "n_non_nan_gex": n_non_nan_gex,
            "genes_std":genes_std,
            "gex_std":gex_std}



metadata_path = os.path.join(data_folder, "metadata.pkl")
with open(metadata_path, "wb") as f:
    pickle.dump(metadata, f)
    
logging.info(f"Metadata saved to {metadata_path}")