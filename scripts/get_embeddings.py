"""Pre-Process Data

Convert the raw data counts into sc-RNAseq compatible data format.

Structure:
    1. Imports, Variables, Functions
    2. Load Data
    3. Convert to `adata` object

"""

# 1. Imports, Variables, Functions
# imports
import numpy as np, os, sys, pandas as pd, scanpy as sc
import anndata as ad
import logging
from tqdm import tqdm
from typing import *
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
from matplotlib import pyplot as plt
from datetime import datetime


# variables
# diseases_of_interest_set = {"Influenza", "Colorectal Carcinoma"}
diseases_of_interest_set = None

example_data_path = (
    "/aloy/home/ddalton/projects/disease_signatures/data/DiSignAtlas/tmp/DSA00123.csv"
)

df_info_path = os.path.join(
    "/aloy",
    "home",
    "ddalton",
    "projects",
    "disease_signatures",
    "data",
    "DiSignAtlas",
    "Disease_information_Datasets_extended.csv",
)


large_df_path = "/aloy/home/ddalton/projects/disease_signatures/data/DiSignAtlas/DiSignAtlas.exp_prof_merged.csv"

base_output_dir = "../data"

# functions
def get_skip_rows(dsaids_interest):
    """Get Skip Rows
    Args:
        - dsaids_interest (list): List of DSAIDs of interest
    Returns:
        skip_rows_idxs (np.array): Array of indexes to skip
    """
    # variables
    large_df_path = "/aloy/home/ddalton/projects/disease_signatures/data/DiSignAtlas/DiSignAtlas.exp_prof_merged.csv"

    # load entire dataframe ID column only
    id_values = pd.read_csv(large_df_path, usecols=["ID"])["ID"].values

    # get indexes to skip
    skip_rows_idxs = np.argwhere(
        ~np.isin([x.split(";")[0] for x in id_values], dsaids_interest)
    ).flatten()

    skip_rows_idxs = skip_rows_idxs + 1  # add 1 to skip

    logging.info(f"Skipping {len(skip_rows_idxs)} rows")
    return skip_rows_idxs


def get_exp_prof(dsaids_interest):
    """Get Expression Profiles"""

    # variables
    file_dir = "/aloy/home/ddalton/projects/disease_signatures/data/DiSignAtlas/tmp/"
    first = True
    for dsaid in tqdm(dsaids_interest):
        __df = pd.read_csv(os.path.join(file_dir, f"{dsaid}.csv"))
        if first:
            df_global = __df
            first = False
        else:
            df_global = pd.concat([df_global, __df], axis=0)
    return df_global


def get_tissue(ids:List[str])->List[str]:
    """Get Tissue
    Args:
        - ids (list): List of IDs
    Returns:
        - tissues (list): List of tissues
    """
    dsaids = [x.split(";")[0] for x in ids]
    dsaid_2_tissue = dict(zip(df_info["dsaid"], df_info["tissue"]))
    tissues = [str(dsaid_2_tissue[dsaid]) for dsaid in dsaids]
    return tissues


def get_disease_study(ids:List[str])->List[str]:
    """Get Disease Study
    Args:
        - ids (list): List of IDs
    Returns:
        - diseases (list): List of diseases
    """
    dsaids = [x.split(";")[0] for x in ids]
    dsaid_2_disease = dict(zip(df_info["dsaid"], df_info["disease"]))
    disease_study = [str(dsaid_2_disease[dsaid]) for dsaid in dsaids]
    return disease_study

def get_disease(ids:List[str])->List[str]:
    """Get Disease
    Args:
        - ids (list): List of IDs
    Returns:
        - diseases (list): List of diseases
    """
    dsaid_2_disease = dict(zip(df_info["dsaid"], df_info["disease"]))
    diseases = list()
    for id in ids:
        dsaid = id.split(";")[0]
        state = id.split(";")[2]
        if state == "Control":
            diseases.append("Control")
        else:
            diseases.append(dsaid_2_disease.get(dsaid))
    return diseases


def get_dataset(ids:List[str])->List[str]:
    """Get Dataset
    Args:
        - ids (list): List of IDs
    Returns:
        - datasets (list): List of datasets
    """
    dsaids = [x.split(";")[0] for x in ids]
    dsaid_2_dataset = dict(zip(df_info["dsaid"], df_info["accession"]))
    datasets = [str(dsaid_2_dataset[dsaid]) for dsaid in dsaids]
    return datasets


def get_folder_name(base_output_dir:str)->str:
    """Get Folder Name
    Args:
        - output_path (str): Output folder
    Returns:
        - output_dir (str): Output directory
    """
    # Step 1: Generate today's date string
    today = datetime.now().strftime("%y-%m-%d")

    # Step 2: Find the highest existing run number for today
    existing_runs = [
        d for d in os.listdir(base_output_dir)
        if os.path.isdir(os.path.join(base_output_dir, d)) and d.startswith(f"ppdata-{today}")
    ]

    # Extract numbers from existing runs and find the max
    existing_numbers = [
        int(d.split("-")[-1]) for d in existing_runs if d.split("-")[-1].isdigit()
    ]

    # Calculate the next run number
    next_run_number = max(existing_numbers, default=0) + 1

    # Step 3: Create the directory name with zero-padded run number
    output_dir = os.path.join(base_output_dir, f"run-{today}-{next_run_number:02d}")

    # Step 4: Create the directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory created: {output_dir}")
    return output_dir

# 2. Load Data
# df = pd.read_csv(example_data_path)
df_info = pd.read_csv(df_info_path)


# Query data to retrieve dsaids of interest
library_strategies_of_interest_set = {"RNA-seq", "Microarray"}

if diseases_of_interest_set :
    QUERY = "disease in @diseases_of_interest_set & library_strategy in @library_strategies_of_interest_set & organism == 'Homo sapiens'"
    dsaids_interest = df_info.query(QUERY)["dsaid"].to_list()
    df = get_exp_prof(dsaids_interest)

else:
    QUERY = "library_strategy in @library_strategies_of_interest_set & organism == 'Homo sapiens'"
    dsaids_interest = np.array(df_info.query(QUERY)["dsaid"].to_list())
    size_df = len(pd.read_csv(large_df_path,usecols=["ID"]))
    
    logging.info(f"Reading merged dataframe {large_df_path}")

    list_filtered_df = list()

    for df_chunk in tqdm(pd.read_csv(large_df_path, chunksize=500), total=int(size_df/500)):
        all_data_ids = df_chunk["ID"].to_list()
        all_data_dsaids = np.array([id.split(";")[0] for id in all_data_ids])
        
        logging.debug(f"all_data_ids: {all_data_ids}")
        logging.debug(f"all_data_dsaids: {all_data_dsaids}")
        
        mask = np.isin(all_data_dsaids,dsaids_interest)
        logging.debug(f"mask {np.sum(mask)} : {mask}")
        df_chunk_filtered = df_chunk[mask]
        
        logging.debug(f"df_chunk_filtered {df_chunk_filtered}")
        
        list_filtered_df.append(df_chunk_filtered)
    
    # merge filtered dataframes
    df = pd.concat(list_filtered_df)    
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    
logging.info(f"Nº of DSAIDs of interest: {len(dsaids_interest)}")


# load dataframe
logging.info(f"Loaded dataframe with shape: {df.shape}")

# Calculate the number of NaNs in each row
nan_counts = df.isna().sum(axis=1)

# # Filter the DataFrame to keep only rows with NaNs less than or equal to 18,000
# df = df[nan_counts <= 18000]
# logging.info(f"Filtered dataframe with shape: {df.shape}")


# Filter out Unknown samples
mask = [False if id.split(";")[2] == "Unknown" else True for id in df.iloc[:, 0].values]
df = df[mask]
logging.info(f"Filtered out Unkowns from dataframe with shape: {df.shape}")

# 3. Convert to `adata` object
# Extract cell identifiers and gene expression data
ids = df.iloc[:, 0]
gene_expression_data = df.iloc[:, 1:].values
gene_names = df.columns[1:]

# Create an AnnData object
adata = ad.AnnData(X=gene_expression_data)

# Add cell and gene metadata
adata.obs["ids"] = ids.values

# gene symbols
adata.var["gene_symbols"] = gene_names

# gene index - nomenclature scGPT
adata.var["index"] = gene_names

# get dataset
datasets = get_dataset(ids)
adata.obs["dataset"] = datasets

# get dsaid
dsaids = [x.split(";")[0] for x in ids]
adata.obs["dsaid"] = dsaids

# get tissues
tissues = get_tissue(ids)
adata.obs["tissue"] = tissues

# get nº genes
n_genes = (~np.isnan(adata.X)).sum(axis=1)
adata.obs["n_genes"] = n_genes

# get disease
diseases = get_disease(ids)
adata.obs["disease"] = diseases

# get disease
diseases = get_disease(ids)
adata.obs["disease_study"] = diseases

# save to output file
output_folder = get_folder_name(base_output_dir)

# save adata
adata.write(os.path.join(output_folder, "data.h5ad"))

# save metadata
if diseases is None:
    metadata = "All Human Diseases"
else:
    metadata = ", ".join(diseases_of_interest_set)

metadata_path = os.path.join(output_folder, "metadata.txt")
with open(metadata_path, "w") as f:
    f.write(metadata)