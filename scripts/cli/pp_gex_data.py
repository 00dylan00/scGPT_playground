"""Pre-Process Data

Convert the raw data counts into sc-RNAseq compatible data format.

Structure:
    1. Imports, Variables, Functions
    2. Load Data
    3. Convert to `adata` object
    4. Save to output file

"""

# region 1. Imports, Variables, Functions
# imports
import numpy as np, os, sys, pandas as pd, scanpy as sc
import anndata as ad
import logging
from tqdm import tqdm
from typing import *
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
from matplotlib import pyplot as plt
from datetime import datetime
import pickle
from typing import *
import json
import xml.etree.ElementTree as ET
import random
sys.path.append("../../")
from tqdm.contrib.concurrent import process_map
from typing import *
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import networkx as nx
import logging
import json
import sys
import pickle
sys.path.append("/aloy/home/ddalton/projects/scGPT_playground/")
from src.preprocessing import pipeline as pp
from src.utils import utils as mu

manual_parameters = { 
    "dataset_exercise":"umls_clean",                 
    "diseases_of_interest_set": None,
    "library_strategies_of_interest_set": ["RNA-Seq", "Microarray"],
    "processing": "linear"  # options: "log2", "scgpt_pp", None
}

# variables
base_output_dir = "../../data"

# functions

# endregion

# region 2. Load Data
diseases_of_interest_set = manual_parameters.get("diseases_of_interest_set")
library_strategies_of_interest_set = manual_parameters.get("library_strategies_of_interest_set")

# load DataFtame Info
df_info = mu.load_dsa_info()

# Load DO data mappings
do_g = mu.load_do_graph()

if manual_parameters.get("dataset_exercise"):
    if manual_parameters["dataset_exercise"] == "umls_clean":
        
        print("UMLS Dataset")
        # get dsaids of interest
        dsaids_interest = mu.get_doids_with_umls(manual_parameters.get("library_strategies_of_interest_set"))
        
        # df = pp.get_processed_exp_prof(dsaids_interest, normalize=manual_parameters.get("normalize"))
        b_p = pp.bulk_processing(processing=manual_parameters.get("processing"), do_z_transform=False, agg_genes="median")

        # process ids
        d_types, df = b_p.get_processed_exp_prof(dsaids_interest)
        ids = df["ID"].to_list()

        logging.info(f"Loaded - Nº total samples: {df.shape}")

        # add information of sample counts!
        df_info = pp.add_sample_counts(ids, df_info)        
        
        # Filter using df_info appropriate dsaids
        df_info = pp.clean_dsaids_qc(df_info, disease_label="diseaseid", n_samples=2, n_dt=2)
        
        # filter expression by passed QC dsaids
        _passed_qc_dsaids = df_info["dsaid"].to_list()
        _dsaids = [x.split(".")[0] for x in ids]
        mask = np.isin(_dsaids, _passed_qc_dsaids)
        df = df[mask].reset_index(drop=True)    # drop previous index
        logging.info(f"QC Filter - Nº total samples: {df.shape}")

# endregion

# region 3. Convert to `adata` object
# Extract cell identifiers and gene expression data
ids = df.iloc[:, 0]
gene_expression_data = df.iloc[:, 1:].values
gene_names = df.columns[1:]

# Create an AnnData object
adata = ad.AnnData(X=gene_expression_data)
adata.X = adata.X.astype(float)


# gene symbols/name
adata.var["gene_symbols"] = gene_names
adata.var["gene_name"] = gene_names
adata.var["index"] = gene_names

# Add cell and gene metadata
adata.obs["ids"] = ids.values

# get dataset
datasets = pp.get_dataset(ids, df_info)
adata.obs["dataset"] = datasets
adata.obs["dataset_id"] = datasets

# get batch
dataset_accessions, batch_ids = pp.get_dataset_to_batch(ids, df_info)
adata.obs["batch"] = batch_ids
adata.obs["batch_id"] = batch_ids

# get dsaid
dsaids = [x.split(".")[0] for x in ids]
adata.obs["dsaid"] = dsaids

# get tissues
tissues = pp.get_tissue(ids, df_info)
adata.obs["tissue"] = tissues

# get nº genes
n_genes = (~np.isnan(adata.X)).sum(axis=1)
adata.obs["n_genes"] = n_genes

# get disease
diseases = pp.get_disease(ids, df_info)
adata.obs["disease"] = diseases
adata.obs["celltype"] = diseases

# get disease
diseases_study = pp.get_disease_study(ids, df_info)
adata.obs["disease_study"] = diseases_study

# get library
library_stratergy = pp.get_library(ids, df_info)
adata.obs["library"] = library_stratergy

if manual_parameters.get("dataset_exercise") == "umls_clean":
    # create dataframe with mapping UMLS & DOID
    _df_info = mu.load_dsa_info()
    _df_info_filtered = _df_info[_df_info["dsaid"].isin(dsaids)]    
    
    # load DO graph
    _do_g = mu.load_do_graph()
    doid_2_term = {
        node: data["name"] for node, data in _do_g.nodes(data=True) if "name" in data
    }

    # Get UMLS mappings to DOIDs
    _uml_2_doid = mu.get_umls_2_doid_mapping(_do_g)
    
    _df_info_filtered["doid"] = _df_info_filtered["diseaseid"].apply(
        lambda x: _uml_2_doid[x]
    )

    # map to dsaids -> doids
    dsaid_2_doids = dict(zip(_df_info_filtered["dsaid"], _df_info_filtered["doid"]))
    
    # store mappings
    doid_study = [dsaid_2_doids.get(id)[0] for id in dsaids]
    doid_disease, doid_id = pp.get_doid_disease(ids, doid_2_term,dsaid_2_doids)
    adata.obs["doid_study"] = doid_study
    adata.obs["doid_id"] = doid_id
    adata.obs["do_id"] = doid_id
    adata.obs["doid_disease"] = doid_disease

# save to output file
output_folder = pp.get_folder_name(base_output_dir)

# endregion

# region 4. Save to output file

# save adata
adata.write(os.path.join(output_folder, "data.h5ad"))

# save metadata
if diseases_of_interest_set is None:
    metadata_txt = "All Human Diseases"
else:
    metadata_txt = ", ".join(diseases_of_interest_set)

# compute metadata values
n_genes = adata.X.shape[1]
n_gex = adata.X.shape[0]    
n_non_nan_genes = np.sum(~np.isnan(adata.X), axis=0)
n_non_nan_gex = np.sum(~np.isnan(adata.X), axis=1)
genes_std = np.nanstd(adata.X, axis=0)
gex_std = np.nanstd(adata.X, axis=1)

# compute nº non-nan values per disease-dataset
all_dis_dt = [ds+";"+dt for ds,dt in zip(diseases_study,datasets)]
unique_dis_dt = list(set(all_dis_dt))
gene_expression_data_bool = ~np.isnan(adata.X)

n_non_nan_dis_dt_row = list()
n_non_nan_dis_dt_col = list()
for dis_dt in tqdm(unique_dis_dt):
    row_mask = np.isin(all_dis_dt,dis_dt)
    
    # get rows of interest
    rows_interest = gene_expression_data_bool[row_mask]

    # merge by columns
    merge_columns = rows_interest.sum(axis=0).astype(bool)

    # get nº non-nan values
    n_non_nan_values = merge_columns.sum()
    
    # append to list
    n_non_nan_dis_dt_row.append(n_non_nan_values)
    n_non_nan_dis_dt_col.append(merge_columns)

n_non_nan_dis_dt_genes = np.array(n_non_nan_dis_dt_col).sum(axis=0)

metadata = {"metadata": metadata_txt,
            "n_genes": n_genes,
            "n_gex": n_gex,
            "n_non_nan_genes": n_non_nan_genes,
            "n_non_nan_gex": n_non_nan_gex,
            "genes_std":genes_std,
            "gex_std":gex_std,
            "unique_dis_dt":unique_dis_dt,
            "n_non_nan_dis_dt_row":n_non_nan_dis_dt_row,
            "n_non_nan_dis_dt_genes":n_non_nan_dis_dt_genes,
            }



metadata_path = os.path.join(output_folder, "metadata.pkl")
with open(metadata_path, "wb") as f:
    pickle.dump(metadata, f)
    
logging.info(f"Metadata saved to {metadata_path}")


# save manual parameters
# Write parameters to a JSON file
with open(os.path.join(output_folder,"parameters.json"), 'w') as json_file:
    json.dump(manual_parameters, json_file, indent=4)


# endregion