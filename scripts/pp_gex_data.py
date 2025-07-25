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
import xml.etree.ElementTree as ET
import random

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
    "dataset_exercise":"doid_dataset",                 
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


mesh_file_path = os.path.join(
        "/aloy/home/ddalton/projects/disease_signatures/data/MeSH/desc2023.xml"
    )


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


def get_mesh_disease(ids:List[str], mesh_id_2_term:dict, dsaid_2_mesh_id:dict)->List[str]:
    """Get MeSH Disease
    Args:
        - ids (list): List of IDs
    Returns:
        - mesh_diseases (list): List of diseases
    """
    mesh_diseases = list()
    mesh_ids = list()
    for id in ids:
        dsaid = id.split(";")[0]
        state = id.split(";")[2]
        if state == "Control":
            mesh_diseases.append("Control")
            mesh_ids.append("Control")
        else:
            mesh_id = dsaid_2_mesh_id.get(dsaid)[0]
            mesh_disease = mesh_id_2_term.get(mesh_id)
            
            mesh_diseases.append(mesh_disease)
            mesh_ids.append(mesh_id)

    return mesh_diseases, mesh_ids


def get_doid_disease(ids:List[str], doid_2_term:dict, dsaid_2_doid:dict)->List[str]:
    """Get DOID Disease
    Args:
        - ids (list): List of IDs
        - doid_2_term (dict): Dictionary with DOID to term
        - dsaid_2_doid (dict): Dictionary with DSAID to DOID
    Returns:
        - doid_diseases (list): List of diseases
    """
    doid_diseases = list()
    doid_ids = list()
    for id in ids:
        dsaid = id.split(";")[0]
        state = id.split(";")[2]
        if state == "Control":
            doid_diseases.append("Control")
            doid_ids.append("Control")
        else:
            doid_id = dsaid_2_doid.get(dsaid)[0]
            doid_disease = doid_2_term.get(doid_id)
            
            doid_diseases.append(doid_disease)
            doid_ids.append(doid_id)

    return doid_diseases, doid_ids


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



def parse_mesh_data_with_ids(file_path):
    """Parse MeSH XML data and extract disease terms along with MeSH IDs."""
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Dictionaries to map tree numbers and MeSH IDs to disease terms
    tree_2_term = {}
    term_2_tree = {}
    mesh_id_2_term = {}

    # Extract disease terms, tree numbers, and MeSH IDs
    for descriptor in root.findall("DescriptorRecord"):
        # Get the disease term name
        term = descriptor.find("DescriptorName/String").text

        # Get the MeSH ID
        mesh_id = descriptor.find("DescriptorUI").text
        mesh_id_2_term[mesh_id] = term

        # Get all tree numbers for this term
        tree_numbers = descriptor.findall("TreeNumberList/TreeNumber")

        for tree_number in tree_numbers:
            # Map each tree number to its term
            tree_2_term[tree_number.text] = term
            term_2_tree[term] = tree_number.text

    return tree_2_term, term_2_tree, mesh_id_2_term




def parse_mesh_data_with_ids(file_path):
    """Parse MeSH XML data and extract disease terms along with MeSH IDs."""
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Dictionaries to map tree numbers and MeSH IDs to disease terms
    tree_2_term = {}
    term_2_tree = {}
    mesh_id_2_term = {}

    # Extract disease terms, tree numbers, and MeSH IDs
    for descriptor in root.findall("DescriptorRecord"):
        # Get the disease term name
        term = descriptor.find("DescriptorName/String").text

        # Get the MeSH ID
        mesh_id = descriptor.find("DescriptorUI").text
        mesh_id_2_term[mesh_id] = term

        # Get all tree numbers for this term
        tree_numbers = descriptor.findall("TreeNumberList/TreeNumber")

        for tree_number in tree_numbers:
            # Map each tree number to its term
            tree_2_term[tree_number.text] = term
            term_2_tree[term] = tree_number.text

    return tree_2_term, term_2_tree, mesh_id_2_term


def shorten_terms(mesh_terms: List) -> List:
    """Shorten Terms
    Args:
        - mesh_terms(List): List of MeSH terms
    Returns:
        - List: List of shortened MeSH terms
    """
    return [x.split(".")[0] for x in mesh_terms]


def get_unrelated_dsaids_from_dataset(dsaids: List, dsaid_2_tree_terms: dict) -> Set:
    """Check if 2+ dsaids from the same dataset are unrelated!
    Args:
        - mesh_id_1(str): MeSH ID 1
        - mesh_id_2(str): MeSH ID 2
        - dsaid_2_tree_terms(dict): Dictionary with MeSH ID to term
    Returns:
        - Set: MeSH IDs that are unrelated
    """
    unrelated_dsaids = set()
    n_dsaids = len(dsaids)
    for i in range(n_dsaids):
        for j in range(i + 1, n_dsaids):
            dsaid_i = dsaids[i]
            dsaid_j = dsaids[j]

            # get tree terms
            term_i = dsaid_2_tree_terms.get(dsaid_i)
            term_j = dsaid_2_tree_terms.get(dsaid_j)

            # exclude overly general terms - i.e don't want "Neoplasm"
            term_i = [x for x in term_i if len(x.split(".")) > 1]
            term_j = [x for x in term_j if len(x.split(".")) > 1]

            # shorten terms to simplify comparisons
            term_i = shorten_terms(term_i)
            term_j = shorten_terms(term_j)

            # check no related terms between
            if len(set(term_i).intersection(set(term_j))) == 0:
                unrelated_dsaids.add(dsaid_i)
                unrelated_dsaids.add(dsaid_j)

    return unrelated_dsaids


def get_unrelated_dsaids_from_all(df: pd.DataFrame, dsaid_2_tree_terms: dict) -> List:
    """Get unrelated dsaids from all datasets
    Args:
        - df(pd.DataFrame): DataFrame with the information
    Returns:
        - List: List of unrelated dsaids
    """
    # Get datasets which have 1+ diseases
    df = df.groupby("accession").filter(lambda x: x["mesh_id"].nunique() >= 2)

    unrelated_dsaids = set()
    for dataset_i in df["accession"].unique():
        QUERY = f"accession == '{dataset_i}'"
        dsaids_i = df.query(QUERY)["dsaid"].to_list()
        unrelated_dsaids_i = get_unrelated_dsaids_from_dataset(
            dsaids_i, dsaid_2_tree_terms
        )
        unrelated_dsaids.update(unrelated_dsaids_i)

    return list(unrelated_dsaids)


def get_bias_dataset(
    df_filtered: pd.DataFrame,
    dsaid_2_tree_terms: dict,
    dsaid_2_mesh_id: dict,
    mesh_id_2_term: dict,
) -> pd.DataFrame:
    """Generate a dataset which allows us to asses dataset biases.
    To do so we select datasets that have 2+ unrelated diseases.
    For completeness we  include for those diseases w/ less than 5 datasets additional datasets.

    Args:
        - df_info(pd.DataFrame): DataFrame with the information
        - dsaid_2_tree_terms(dict): Dictionary with MeSH ID to term
        - dsaid_2_mesh_id(dict): Dictionary with MeSH ID to term
        - mesh_id_2_term(dict): Dictionary with MeSH ID to term
    Returns:
        - pd.DataFrame: DataFrame with the information
    """

    logging.info(df_info.shape)

    # filter by nº of samples
    n_samples_upp_thr = 200
    n_samples_low_thr = 10

    # Filter by nº of samples
    # add nº of samples - must have both control & case
    df_filtered = df_filtered.copy()
    df_filtered["n_samples"] = [
        (
            len(x.split(";")) + len(y.split(";"))
            if isinstance(x, str) and isinstance(y, str)
            else 0
        )
        for x, y in zip(df_filtered["Control"], df_filtered["Case"])
    ]

    # filter
    QUERY = f"n_samples >= {n_samples_low_thr} & n_samples <= {n_samples_upp_thr}"
    df_query = df_filtered.query(QUERY)
    logging.info(f"Filter: nº of samples: {df_query.shape[0]}")

    # Filter by presence of Disease MeSH IDs
    # mask
    mask_mesh_tree = list()
    for dsaid_i in df_query["dsaid"]:
        mesh_tree_i = dsaid_2_tree_terms.get(dsaid_i)
        if mesh_tree_i is not None:
            # presence of disease tree term
            mesh_tree_i = [x for x in mesh_tree_i if x.startswith("C")]
            if len(mesh_tree_i) > 0:
                mask_mesh_tree.append(True)
            else:
                mask_mesh_tree.append(False)
        else:
            mask_mesh_tree.append(False)

    df_query = df_query[mask_mesh_tree]
    logging.info(f"Filter: MeSH Tree Disease presence {df_query.shape[0]}")

    # add mesh id
    df_query["mesh_id"] = [
        dsaid_2_mesh_id.get(x)[0] if len(dsaid_2_mesh_id.get(x)) > 0 else np.nan
        for x in df_query["dsaid"]
    ]
    df_query = df_query.dropna(subset=["mesh_id"])
    logging.info(f"Filter by mesh_ids {df_query.shape[0]}")

    # add mesh id info
    df_query["mesh_disease"] = [mesh_id_2_term.get(x) for x in df_query["mesh_id"]]

    # Filter diseases present in 5+ datasets
    # Group by disease
    disease_counts = df_query.groupby("mesh_id")["accession"].nunique()

    # Filter for diseases that are in 5 or more unique datasets
    disease_counts = disease_counts[disease_counts >= 5]
    df_query = df_query[df_query["mesh_id"].isin(disease_counts.index)]
    logging.info(f"Filter 5+ datasets for each disease: {df_query.shape[0]}")

    # Get dsaids with another dsaid from an unrelated disease in the same dataset
    unrelated_dsaids = get_unrelated_dsaids_from_all(df_query, dsaid_2_tree_terms)
    logging.info(f"Nº of unrelated dsaids {len(unrelated_dsaids)}")

    # filter df_query_2 by unrelated dsaids
    df_unrelated = df_query[df_query["dsaid"].isin(unrelated_dsaids)]
    logging.info(f"Filter: Unrelated diseases {df_unrelated.shape[0]}")

    # filter out all datasets that have unrelated diseases
    df_rest = df_query[~df_query["accession"].isin(df_unrelated["accession"])]

    # only have 1 dsaid for each disease
    df_rest = df_rest.copy()
    df_rest["mesh_id_accession"] = [
        x + "_" + y for x, y in zip(df_rest["mesh_id"], df_rest["accession"])
    ]
    df_rest.drop_duplicates(subset=["mesh_id_accession"], inplace=True)

    # check which diseases have less than 5 datasets/accessions
    dsaids_rest = set()
    for mesh_id_i in df_unrelated["mesh_id"].unique():
        accessions_i = df_unrelated.query(f"mesh_id == '{mesh_id_i}'")[
            "accession"
        ].unique()
        n_datasets_i = len(accessions_i)
        if n_datasets_i < 5:
            n_samples_i = 5 - n_datasets_i
            dsaids_i = df_rest.query(f"mesh_id == '{mesh_id_i}'")["dsaid"].to_list()
            remaining_samples_i = set(dsaids_i) - dsaids_rest
            sample_i = random.sample(list(remaining_samples_i), n_samples_i)
            dsaids_rest.update(sample_i)
        else:
            logging.info(f"MeSH id {mesh_id_i} has {n_datasets_i} datasets")

    logging.info(f"DSAIDs to sample {len(dsaids_rest)}")

    # get dsaids for diseases with less than 5
    df_final = pd.concat([df_unrelated, df_rest.query("dsaid in @dsaids_rest")])
    logging.info(f"Final shape {df_final.shape[0]}")
    return df_final

def get_data_leakage_dataset(
    df_filtered: pd.DataFrame,
    dsaid_2_tree_terms: dict,
    dsaid_2_mesh_id: dict,
    mesh_id_2_term: dict,
    thr_n_datasets=20,
) -> pd.DataFrame:
    """Generate a dataset which allows us to asses how much our model can generalize to new datasets.
    To do so we select diseases which appear in 20+ datasets..

    Args:
        - df_info(pd.DataFrame): DataFrame with the information
        - dsaid_2_tree_terms(dict): Dictionary with MeSH ID to term
        - dsaid_2_mesh_id(dict): Dictionary with MeSH ID to term
        - mesh_id_2_term(dict): Dictionary with MeSH ID to term
    Returns:
        - pd.DataFrame: DataFrame with the information
    """

    logging.info(df_info.shape)

    # filter by nº of samples
    n_samples_upp_thr = 100
    n_samples_low_thr = 10

    # Filter by nº of samples
    # add nº of samples - must have both control & case
    df_filtered = df_filtered.copy()
    df_filtered["n_samples"] = [
        (
            len(x.split(";")) + len(y.split(";"))
            if isinstance(x, str) and isinstance(y, str)
            else 0
        )
        for x, y in zip(df_filtered["Control"], df_filtered["Case"])
    ]

    # filter
    QUERY = f"n_samples >= {n_samples_low_thr} & n_samples <= {n_samples_upp_thr}"
    df_query = df_filtered.query(QUERY)
    logging.info(f"Filter: nº of samples: {df_query.shape[0]}")

    # Filter by presence of Disease MeSH IDs
    # mask
    mask_mesh_tree = list()
    for dsaid_i in df_query["dsaid"]:
        mesh_tree_i = dsaid_2_tree_terms.get(dsaid_i)
        if mesh_tree_i is not None:
            # presence of disease tree term
            mesh_tree_i = [x for x in mesh_tree_i if x.startswith("C")]
            if len(mesh_tree_i) > 0:
                mask_mesh_tree.append(True)
            else:
                mask_mesh_tree.append(False)
        else:
            mask_mesh_tree.append(False)

    df_query = df_query[mask_mesh_tree]
    logging.info(f"Filter: MeSH Tree Disease presence {df_query.shape[0]}")

    # add mesh id
    df_query["mesh_id"] = [
        dsaid_2_mesh_id.get(x)[0] if len(dsaid_2_mesh_id.get(x)) > 0 else np.nan
        for x in df_query["dsaid"]
    ]
    df_query = df_query.dropna(subset=["mesh_id"])
    logging.info(f"Filter by mesh_ids {df_query.shape[0]}")

    # add mesh id info
    df_query["mesh_disease"] = [mesh_id_2_term.get(x) for x in df_query["mesh_id"]]

    # Filter diseases present in n+ datasets
    # Group by disease
    disease_counts = df_query.groupby("mesh_id")["accession"].nunique()

    # Filter for diseases that are in n+ unique datasets
    disease_counts = disease_counts[disease_counts >= thr_n_datasets]
    df_final = df_query[df_query["mesh_id"].isin(disease_counts.index)]
    logging.info(
        f"Filter {thr_n_datasets}+ datasets for each disease: {df_query.shape[0]}"
    )

    logging.info(f"Nº unique diseases: {df_final['mesh_id'].nunique()}")
    logging.info(f"Nº unique datasets: {df_final['accession'].nunique()}")

    return df_final

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
import obonet


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",

)

# variables
data_info_path = os.path.join(
    "/aloy/home/ddalton/projects/disease_signatures",
    "data",
    "DiSignAtlas",
    "Disease_information_Datasets.csv",
)

do_obl_data_path = os.path.join(
    "/aloy/home/ddalton/projects/disease_signatures",
    "data",
    "DiseaseOntology",
    "doid.obo",
)
external_links_data_path = os.path.join(
    "/aloy/home/ddalton/projects/disease_signatures",
    "data",
    "DiSignAtlas",
    "external_links.pkl",
)


# functions

def load_do_graph():
    """Load the Disease Ontology graph from the OBO file."""
    do_obl_data_path = os.path.join(
        "/aloy/home/ddalton/projects/disease_signatures",
        "data",
        "DiseaseOntology",
        "doid.obo",

    )
    do_graph = obonet.read_obo(do_obl_data_path)
    return nx.DiGraph(do_graph)


def get_processed_ids():
    """Get processed ids
    Returns:
        list: list of processed ids
    """
    data_path = os.path.join(
        "/aloy/home/ddalton/projects/disease_signatures",
        "data",
        "DiSignAtlas",
        "dsa_diff_download.processed",
    )
    return [f.split("_")[0] for f in os.listdir(data_path)]


# get all entrez protein-coding human ids
def get_human_entrez_protein_coding_ids():
    """Get Human Entrez IDs
    Returns:
        list: list of human entrez ids
    """
    data_path = os.path.join(
        "/aloy/home/ddalton/projects/disease_signatures",
        "data",
        "ncbi_gene_info",
        "gene_info",
    )
    df = pd.read_csv(data_path, sep="\t", usecols=["#tax_id", "GeneID", "type_of_gene"])
    df_human = df[(df["#tax_id"] == 9606) & (df["type_of_gene"] == "protein-coding")]
    logging.info(f"Nº of Human protein coding genes: {len(df_human)}")
    return df_human["GeneID"].to_list()

def get_de_genes(
    signature: List, thr_log2FC: float = 0, thr_p_val: float = 0.05,
) -> tuple[List, List]:
    """
    Get DE genes for each signature
    Args:
        - thr_log2FC: float
            Threshold for log2 fold change
        - thr_p_val: float
            Threshold for p-value

    Returns:
        - upregulated genes: list
        - downregulated genes: list
    """
    genes = np.array(signature[2])
    log2fc = np.array(signature[5])
    p_values = np.array(signature[4])

    global human_entrez_protein_coding_ids

    # filter by human genes
    mask_human = np.isin(np.array(genes), human_entrez_protein_coding_ids)
    genes = np.array(genes)[mask_human]
    p_values = np.array(p_values)[mask_human]
    log2fc = np.array(signature[5])[mask_human]
    total_genes = len(genes)

    # filter by thresholds
    mask_upr = (log2fc >= thr_log2FC) & (p_values <= thr_p_val)
    mask_dwn = (log2fc <= -thr_log2FC) & (p_values <= thr_p_val)

    return genes[mask_upr], genes[mask_dwn], total_genes



def get_dataset_do():
    """
    This fuction compiles filtering steps to get final list of dsaids
    """

    # variables
    path_pkl = os.path.join(
        "/aloy/home/ddalton/projects/disease_signatures/",
        "data",
        "DiSignAtlas",
        "signatures.pkl",
    )


    # Load data

    processed_ids = get_processed_ids()
    logging.info(f"Nº of processed ids: {len(processed_ids)}")

    df_data_info = pd.read_csv(data_info_path)

    df_data_info_processed = df_data_info.copy()

    df_data_info_processed = df_data_info_processed[
        df_data_info_processed["dsaid"].isin(processed_ids)
    ]

    logging.info(f"Nº of processed ids in df_data_info: {len(df_data_info_processed)}")

    df_data_info_processed_filtered = df_data_info_processed[
        (df_data_info_processed["organism"] == "Homo sapiens")
        & (
            (df_data_info_processed["library_strategy"] == "Microarray")
            | (df_data_info_processed["library_strategy"] == "RNA-Seq")
        )
    ]
    logging.info(
        f"Nº of Filtered by library (filter out single cell): {df_data_info_processed_filtered.shape}"
    )

    # get all entrez protein-coding human ids
    global human_entrez_protein_coding_ids

    logging.info(f"Loading signatures from file {path_pkl}")
    signatures = pickle.load(open(path_pkl, "rb"))


    # load disease ontology graph
    do_G = load_do_graph()

    # load external links
    with open(external_links_data_path, "rb") as f:
        external_links = pickle.load(f)
    print(f"Loaded {len(external_links)} DiSignAtlas metadata (external links)")

    # get DOID for each dsa
    dsaids_2_doids = {}
    for dsaid, external_link in zip(
        external_links["dsaids"], external_links["external_links"]
    ):
        do_ids = [
            link.replace("DO:", "DOID:") for link in external_link if link.startswith("DO")
        ]
        if len(do_ids) > 0:
            dsaids_2_doids[dsaid] = do_ids

    dsaids_with_doid = list(dsaids_2_doids.keys())

    # Filter 1: Human Genes
    human_dsaids = df_data_info_processed_filtered["dsaid"].to_list()

    pct_thr = 0.5
    # even though we have filtered by human, we now look at the actual genes! Some may not be human !
    pct_human = list()
    dsaids_f1 = list()
    for i in range(len(signatures)):
        if signatures[i][0] in human_dsaids:
            _pct = len(
                set(signatures[i][2]).intersection(human_entrez_protein_coding_ids)
            ) / len(set(signatures[i][2]))
            pct_human.append(_pct)
            if _pct > pct_thr:
                dsaids_f1.append(signatures[i][0])
    pct_human = np.array(pct_human)


    df_query = df_data_info_processed_filtered.query("dsaid in @dsaids_f1")
    print(f"Nº of signatures: {len(df_query)}")
    print(f"Nº of datasets : {len(df_query['accession'].unique())}")
    print(f"Nº of diseases : {len(df_query['disease'].unique())}")


    # Filter 2: Type of Sequencing
    df_data_info_processed_filtered["library_strategy"].value_counts()
    dsaids_f2 = dsaids_f1

    # Filter 3: Presence of Disease Ontology IDs
    # Leaf nodes Diseases
    leaf_nodes = [
        node
        for node in do_G.nodes()
        if do_G.in_degree(node) == 0 and do_G.out_degree(node) > 0
    ]

    dsaids_w_doids = [d for d in dsaids_f2 if d in dsaids_with_doid]
    dsaids_w_diseases = [
        d for d in dsaids_w_doids if len(set(dsaids_2_doids[d]) & set(leaf_nodes)) > 0
    ]

    # for now only take DSAIDS with 1 LEAF DISEASE!
    dsaids_f3 = [
        d for d in dsaids_w_diseases if len(set(dsaids_2_doids[d]) & set(leaf_nodes)) == 1
    ]
    print(f"Nº of DSAIDS with 1 leaf disease {len(dsaids_f3)}")


    # Filter 3.1: Presence of Disease Ontology IDs
    flatten = lambda l: [item for sublist in l for item in sublist]
    df_query = df_data_info_processed_filtered.query("dsaid in @dsaids_with_doid")
    print(f"Filtering 3.1 - has doid")
    print(f"Nº of signatures: {len(df_query)}")
    print(f"Nº of datasets : {len(df_query['accession'].unique())}")
    print(f"Nº of diseases : {len(df_query['disease'].unique())}")
    print(
        f"Nº of disease ontology ids: {len(set(flatten([dsaids_2_doids[x] for x in dsaids_with_doid])))}"
    )
    print(
        f"Nº of disease ontology diseases: {len(set(flatten([list(set(dsaids_2_doids[x])&set(leaf_nodes)) for x in dsaids_with_doid])))}"
    )

    # Filter 3.2: Presence of Leaf Disease Ontology IDs
    df_query = df_data_info_processed_filtered.query("dsaid in @dsaids_w_diseases")
    print(f"Filtering 3.2 - has doid leaf")
    print(f"Nº of signatures: {len(df_query)}")
    print(f"Nº of datasets : {len(df_query['accession'].unique())}")
    print(f"Nº of diseases : {len(df_query['disease'].unique())}")
    print(
        f"Nº of disease ontology ids: {len(set(flatten([dsaids_2_doids[x] for x in dsaids_w_diseases])))}"
    )
    print(
        f"Nº of disease ontology diseases: {len(set(flatten([list(set(dsaids_2_doids[x])&set(leaf_nodes)) for x in dsaids_w_diseases])))}"
    )

    # Filter 3.3: Presence of 1 Leaf Disease Ontology IDs
    df_query = df_data_info_processed_filtered.query("dsaid in @dsaids_f3")
    print(f"Filtering 3.3 - has 1 doid leaf")
    print(f"Nº of signatures: {len(df_query)}")
    print(f"Nº of datasets : {len(df_query['accession'].unique())}")
    print(f"Nº of diseases : {len(df_query['disease'].unique())}")
    print(
        f"Nº of disease ontology ids: {len(set(flatten([dsaids_2_doids[x] for x in dsaids_f3])))}"
    )
    print(
        f"Nº of disease ontology diseases: {len(set(flatten([list(set(dsaids_2_doids[x])&set(leaf_nodes)) for x in dsaids_f3])))}"
    )

    # Filter 4: Presence of DE Genes
    f_signatures = [s for s in signatures if s[0] in dsaids_f3]
    de_genes = process_map(get_de_genes, f_signatures, max_workers=8, chunksize=10)

    # Filter 4.1: Presence of DE Genes
    # Less than 50% of genes DE
    # at least 50 DE genes
    mask_de_genes = np.array(
        [
            (
                True
                if 50<= len(d[0]) + len(d[1]) <= 0.5 * d[2]
                else False
            )
            for d in de_genes
        ]
    )

    # get dsaids which pass filter
    dsaids_f4 = np.array(dsaids_f3)[mask_de_genes]
    print(f"Filtered by nº of DE genes: {len(dsaids_f4)}")

    df_query = df_data_info_processed_filtered.query("dsaid in @dsaids_f4")
    print(f"Filtering 4 - has 1 doid leaf")
    print(f"Nº of signatures: {len(df_query)}")
    print(f"Nº of datasets : {len(df_query['accession'].unique())}")
    print(f"Nº of diseases : {len(df_query['disease'].unique())}")
    print(
        f"Nº of disease ontology ids: {len(set(flatten([dsaids_2_doids[x] for x in dsaids_f4])))}"
    )
    print(
        f"Nº of disease ontology diseases: {len(set(flatten([list(set(dsaids_2_doids[x])&set(leaf_nodes)) for x in dsaids_f4])))}"
    )

    return dsaids_f4
# endregion

# region 2. Load Data
diseases_of_interest_set = manual_parameters.get("diseases_of_interest_set")
library_strategies_of_interest_set = manual_parameters.get("library_strategies_of_interest_set")

# load DataFtame Info
df_info = pd.read_csv(df_info_path)


# Load MeSH data mappings
tree_2_term, term_2_tree, mesh_id_2_term = parse_mesh_data_with_ids(
    file_path=mesh_file_path
)

mesh_terms = pickle.load(
    open(
        "/aloy/home/ddalton/projects/disease_signatures/data/DiSignAtlas/mesh_tree_terms.pkl",
        "rb",
    )
)
dsaid_2_mesh_id = dict(zip(mesh_terms["dsaids"], mesh_terms["mesh_ids"]))
dsaid_2_tree_terms = dict(zip(mesh_terms["dsaids"], mesh_terms["mesh_tree_terms"]))


# Load DO data mappings
url = "http://purl.obolibrary.org/obo/doid.obo"
do_graph = obonet.read_obo(url)
do_G = nx.DiGraph(do_graph)

# Generate a mapping of DOID to its term (disease name)
doid_2_term = {
    node: data["name"] for node, data in do_G.nodes(data=True) if "name" in data
}

# get leaf nodes
leaf_nodes = [
    node
    for node in do_G.nodes()
    if do_G.in_degree(node) == 0 and do_G.out_degree(node) > 0
]

external_links_data_path = os.path.join(
    "/aloy/home/ddalton/projects/disease_signatures",
    "data",
    "DiSignAtlas",
    "external_links.pkl",
)

# load external links
with open(external_links_data_path, "rb") as f:
    external_links = pickle.load(f)
print(f"Loaded {len(external_links)} DiSignAtlas metadata (external links)")

# get DOID for each dsa
dsaid_2_doids = {}
for dsaid, external_link in zip(
    external_links["dsaids"], external_links["external_links"]
):
    do_ids = [
        link.replace("DO:", "DOID:") for link in external_link if link.startswith("DO")
    ]
    if len(do_ids) > 0:
        dsaid_2_doids[dsaid] = do_ids


# filter by library strategy
QUERY = f"library_strategy in @library_strategies_of_interest_set & organism == 'Homo sapiens'"
df_filtered = df_info.query(QUERY)

sys.exit(1)

if manual_parameters.get("dataset_exercise"):
    if manual_parameters["dataset_exercise"] == "small":
        print("Small Dataset")

    elif manual_parameters["dataset_exercise"] == "medium":
        print("Medium Dataset")

        # get dsaids of interest
        diseases_interest, dsaids_interest = get_medium_dataset(df_filtered, 10)

        df = get_exp_prof(dsaids_interest)


    elif manual_parameters["dataset_exercise"] == "data_leakage":
        print("Data Leakage Dataset")

        df_dl_dataset = get_data_leakage_dataset(
            df_filtered, dsaid_2_tree_terms, dsaid_2_mesh_id, mesh_id_2_term
        )

        dsaids_interest = df_dl_dataset["dsaid"].to_list()

        df = get_exp_prof(dsaids_interest)

    elif manual_parameters["dataset_exercise"] == "bias_dataset":

        # get dsaids of interest
        df_bias = get_bias_dataset(df_filtered, dsaid_2_tree_terms, dsaid_2_mesh_id, mesh_id_2_term)
        dsaids_interest = list(df_bias["dsaid"].unique())

        df = get_exp_prof(dsaids_interest)


    elif manual_parameters["dataset_exercise"] == "large":
        print("Large Dataset")
    elif manual_parameters["dataset_exercise"] == "doid_dataset":
        print("DoID Dataset")

        # load global variables
        human_entrez_protein_coding_ids = get_human_entrez_protein_coding_ids()
        dsaids_interest = get_dataset_do()
        df = get_exp_prof(dsaids_interest)



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


if manual_parameters.get("ontology") == "mesh_id":
    # get mesh_ids
    mesh_id_study = [dsaid_2_mesh_id.get(id)[0] for id in dsaids]
    mesh_disease, mesh_id = get_mesh_disease(ids, mesh_id_2_term, dsaid_2_mesh_id)
    adata.obs["mesh_id_study"] = mesh_id_study
    adata.obs["mesh_id"] = mesh_id
    adata.obs["mesh_disease"] = mesh_disease

elif manual_parameters.get("ontology") == "doid":
    # get doid ids
    doid_study = [dsaid_2_doids.get(id)[0] for id in dsaids]
    doid_disease, doid_id = get_doid_disease(ids, dsaid_2_doids)
    adata.obs["doid_study"] = doid_study
    adata.obs["doid_id"] = doid_id
    adata.obs["doid_disease"] = doid_disease

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