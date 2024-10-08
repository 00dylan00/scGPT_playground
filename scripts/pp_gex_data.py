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

# variables
# manual_parameters = { "diseases_of_interest_set": list({
#     "Colorectal Carcinoma",
#     "Breast Cancer",
#     "Prostate Cancer",
#     "Hepatocellular Carcinoma",
#     "Crohn's Disease",
#     "Multiple Sclerosis"
    
# }),
#     "library_strategies_of_interest_set": list({
#         "Microarray"
#     }),
# }

manual_parameters = { 
    "dataset_exercise":"medium",                 
    "diseases_of_interest_set": None,
    "library_strategies_of_interest_set": list({"RNA-Seq", "Microarray"}),
}

# library_strategies_of_interest_set = {"RNA-Seq", "Microarray"}


# example_data_path = (
#     "/aloy/home/ddalton/projects/disease_signatures/data/DiSignAtlas/tmp/DSA00123.csv"
# )

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


def get_library(ids:List[str])->List[str]:
    """Get Library
    Args:
        - ids (list): List of IDs
    Returns:
        - datasets (list): List of datasets
    """
    dsaids = [x.split(";")[0] for x in ids]
    dsaid_2_dataset = dict(zip(df_info["dsaid"], df_info["library_strategy"]))
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
        if os.path.isdir(os.path.join(base_output_dir, d)) and d.startswith(f"pp_data-{today}")
    ]

    # Extract numbers from existing runs and find the max
    existing_numbers = [
        int(d.split("-")[-1]) for d in existing_runs if d.split("-")[-1].isdigit()
    ]

    # Calculate the next run number
    next_run_number = max(existing_numbers, default=0) + 1

    # Step 3: Create the directory name with zero-padded run number
    output_dir = os.path.join(base_output_dir, f"pp_data-{today}-{next_run_number:02d}")

    # Step 4: Create the directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory created: {output_dir}")
    return output_dir

def get_dataset_to_batch(ids:List[str], df_info:pd.DataFrame)->Tuple[List[str], List[int]]:
    """Get Dataset to Batch
    Args:
        - ids (list): List of IDs
        - df_info (pd.DataFrame): DataFrame with information
    Returns:
        - dataset_accessions (list): List of dataset accessions
        - dataset_ids (list): List of dataset IDs
    """
    dsaid_2_accession = dict(zip(df_info["dsaid"], df_info["accession"]))

    dataset_accessions = [dsaid_2_accession[id.split(";")[0]] for id in ids]

    accession_2_id = {k: v for v, k in enumerate(set(dataset_accessions))}
    dataset_ids = [accession_2_id[accession] for accession in dataset_accessions]

    return dataset_accessions, dataset_ids


def get_diseases_n_datasets(df: pd.DataFrame, n: int = 10) -> List:
    """Get Diseases With More Than n Datasets

    Args:
        - df(pd.DataFrame): DataFrame with the information
        - diseases(List): List of diseases to filter
        - n(int): Number of datasets to filter

    Returns:
        - List: List of diseases with more than n datasets
    """
    diseases_list = list()
    dsaids_list = list()
    # iterate over diseases
    for disease in df["disease"].unique():
        df_query = df.query(f'disease == "{disease}"')
        if df_query["accession"].nunique() >= n:
            diseases_list.append(disease)
            dsaids_list.append(df_query["dsaid"].unique())

    return diseases_list, dsaids_list


def get_medium_dataset(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Get Medium Dataset

    Args:
        - df(pd.DataFrame): DataFrame with the information
        - n(int): Number of datasets to filter

    Returns:
        - pd.DataFrame: DataFrame with the medium dataset
    """
    # load mappings dsaids -> MeSH terms
    mesh_terms = pickle.load(
        open(
            "/aloy/home/ddalton/projects/disease_signatures/data/DiSignAtlas/mesh_tree_terms.pkl",
            "rb",
        )
    )

    dsaid_2_mesh = {
        k: v for k, v in zip(mesh_terms["dsaids"], mesh_terms["mesh_tree_terms"])
    }

    dsaids_with_mesh = [k for k, v in dsaid_2_mesh.items() if len(v) > 0]
    
    
    # filter by nº of datasets
    diseases_list, dsaids_list = get_diseases_n_datasets(df, n)

    # filter by MeSH term presence
    diseases_f_mesh = list()
    dsaids_f_mesh = list()
    for disease_i, dsaids_i in tqdm(zip(diseases_list, dsaids_list),total=len(diseases_list)):
        
        if np.isin(dsaids_i, dsaids_with_mesh).any():
            diseases_f_mesh.append(disease_i)
            for dsaid_j in dsaids_i:
                if dsaid_j in dsaids_with_mesh:
                    dsaids_f_mesh.append(dsaid_j)
                else:
                    logging.info("DSAID of a disease w/ other DSAID w/ MeSH terms - but it itself doesn't have MeSH terms")
                    logging.info(f"{dsaid_j}, {disease_i}")
                                    
    logging.info(f"Nº of diseases {len(diseases_f_mesh)}/{len(diseases_list)}")             
    logging.info(f"Nº of dsaids {len(dsaids_f_mesh)}/{len([x for sublist in dsaids_list for x in sublist])}")             
    return diseases_f_mesh, dsaids_f_mesh

# endregion

# region 2. Load Data
diseases_of_interest_set = manual_parameters.get("diseases_of_interest_set")
library_strategies_of_interest_set = manual_parameters.get("library_strategies_of_interest_set")

# df = pd.read_csv(example_data_path)
df_info = pd.read_csv(df_info_path)


if manual_parameters.get("dataset_exercise"):
    if manual_parameters["dataset_exercise"] == "small":
        print("Small Dataset")

    elif manual_parameters["dataset_exercise"] == "medium":
        print("Medium Dataset")

        # filter by library strategy
        QUERY = f"library_strategy in @library_strategies_of_interest_set"
        df_filtered = df_info.query(QUERY)

        # get dsaids of interest
        diseases_interest, dsaids_interest = get_medium_dataset(df_filtered, 10)

        df = get_exp_prof(dsaids_interest)

    elif manual_parameters["dataset_exercise"] == "large":
        print("Large Dataset")

# if specific diseases
else:
    QUERY = "disease in @diseases_of_interest_set & library_strategy in @library_strategies_of_interest_set & organism == 'Homo sapiens'"
    dsaids_interest = df_info.query(QUERY)["dsaid"].to_list()
    # df = get_exp_prof(dsaids_interest)
    df = get_exp_prof(dsaids_interest)

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



# endregion

# region 3. Convert to `adata` object
# Extract cell identifiers and gene expression data
ids = df.iloc[:, 0]
gene_expression_data = df.iloc[:, 1:].values
gene_names = df.columns[1:]

# Create an AnnData object
adata = ad.AnnData(X=gene_expression_data)

# Add cell and gene metadata
adata.obs["ids"] = ids.values

# gene symbols/name
adata.var["gene_symbols"] = gene_names
adata.var["gene_name"] = gene_names

# gene index - nomenclature scGPT
adata.var["index"] = gene_names

# get dataset
datasets = get_dataset(ids)
adata.obs["dataset"] = datasets

# get dataset
datasets = get_dataset(ids)
adata.obs["dataset_id"] = datasets

# get batch
dataset_accessions, batch_ids = get_dataset_to_batch(ids, df_info)
adata.obs["batch"] = batch_ids
adata.obs["batch_id"] = batch_ids

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

# get celltype
diseases = get_disease(ids)
adata.obs["celltype"] = diseases

# get disease
diseases_study = get_disease_study(ids)
adata.obs["disease_study"] = diseases_study

# get library
library_stratergy = get_library(ids)
adata.obs["library"] = library_stratergy


# save to output file
output_folder = get_folder_name(base_output_dir)



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
gene_expression_data_bool = ~np.isnan(gene_expression_data)

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




# diseases_of_interest_set = {"Influenza", "Colorectal Carcinoma", "Asthma"}
# diseases_of_interest_set = None
# diseases_of_interest_set = {"Huntington's Disease", "Alzheimer's Disease", 'Asthma', 'COVID-19',
#        'Influenza', "Parkinson's Disease", 'Systemic Lupus Erythematosus',
#        'Obesity', 'Hepatocellular Carcinoma', "Crohn's Disease",
#        'Ulcerative Colitis', 'Sepsis', 'Breast Cancer', 'Psoriasis',
#        'Schizophrenia', 'Multiple Sclerosis', 'Amyotrophic Lateral Sclerosis',
#        'Tuberculosis', 'Chronic Obstructive Pulmonary Disease',
#        'Rheumatoid Arthritis', 'Idiopathic Pulmonary Fibrosis',
#        'Colorectal Carcinoma', 'Type 1 Diabetes',
#        'Non-Alcoholic Steatohepatitis', 'Melanoma', 'Diabetes',
#        'Myocardial Infarction', 'Acute Myeloid Leukemia (Aml-M2)', 'Colitis',
#        'Prostate Cancer'}

# diseases_of_interest_set = {'Acute-On-Chronic Liver Failure',
#  "Barrett's Esophagus",
#  "Behcet's Disease",
#  'Chronic Rhinosinusitis',
#  'Cornelia De Lange Syndrome',
#  'Coronary Artery Disease',
#  'Diabetes',
#  'Diabetic Kidney Disease',
#  'Follicular Lymphoma',
#  'Glioblastoma Multiforme',
#  'Hepatitis B',
#  'Hutchinson-Gilford Progeria Syndrome',
#  'Hypertension',
#  'Multiple System Atrophy',
#  'Pneumonia',
#  'Primary Myelofibrosis',
#  'Spinal Muscular Atrophy',
#  'Squamous Cell Carcinoma',
#  'Steatosis',
#  'Type 2 Diabetes Mellitus'}

# diseases_of_interest_set 
# = {'Breast Cancer', 'Colorectal Carcinoma', 'Influenza'}
# diseases_of_interest_set = {'Control', 'Lung Adenocarcinoma', 'Breast Cancer', 'Psoriasis', 'Ulcerative Colitis', "Crohn's Disease", 'Lung Cancer'}

# diseases_of_interest_set = {
#     "Crohn's Disease",
#     "Ulcerative Colitis",
#     "Lung Cancer",
#     "Lung Adenocarcinoma",
#     "Breast Cancer",
#     "Psoriasis",
# }



    #! LARGE DATASET - OLD
    # QUERY = "library_strategy in @library_strategies_of_interest_set & organism == 'Homo sapiens'"
    # dsaids_interest = np.array(df_info.query(QUERY)["dsaid"].to_list())
    # size_df = len(pd.read_csv(large_df_path,usecols=["ID"]))
    
    # logging.info(f"Reading merged dataframe {large_df_path}")

    # list_filtered_df = list()

    # for df_chunk in tqdm(pd.read_csv(large_df_path, chunksize=500), total=int(size_df/500)):
    #     all_data_ids = df_chunk["ID"].to_list()
    #     all_data_dsaids = np.array([id.split(";")[0] for id in all_data_ids])
        
    #     logging.debug(f"all_data_ids: {all_data_ids}")
    #     logging.debug(f"all_data_dsaids: {all_data_dsaids}")
        
    #     mask = np.isin(all_data_dsaids,dsaids_interest)
    #     logging.debug(f"mask {np.sum(mask)} : {mask}")
    #     df_chunk_filtered = df_chunk[mask]
        
    #     logging.debug(f"df_chunk_filtered {df_chunk_filtered}")
        
    #     list_filtered_df.append(df_chunk_filtered)
    
    # # merge filtered dataframes
    # df = pd.concat(list_filtered_df)    
    # if "Unnamed: 0" in df.columns:
    #     df.drop(columns=["Unnamed: 0"], inplace=True)