from datetime import datetime
import pandas as pd, scanpy as sc
from tqdm import tqdm
import os
import numpy as np
import anndata as ad, yaml
from typing import *
from tqdm.contrib.concurrent import thread_map, process_map
from tqdm.contrib.concurrent import thread_map, process_map
import scipy
import numpy as np
from typing import *
from functools import partial
from tqdm.contrib.concurrent import thread_map



def convert_to_adata(df:pd.DataFrame, normalize:bool)-> sc.AnnData:
    """Convert DataFrame to AnnData"""
    # extract IDs, gene expression data, and gene names
    ids = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    gene_names = df.columns[1:]
        
    adata = sc.AnnData(X)
    adata.obs["ids"] = ids
    adata.var["gene_symbols"]= gene_names
    adata.var["gene_name"]= gene_names
    adata.var["index"]= gene_names

    if normalize:
        # get mask of nan values
        nan_mask = np.isnan(X)

        # replace nans with 0 for normalization
        adata.X = np.nan_to_num(adata.X, nan=0.0)

        # CPM
        sc.pp.normalize_total(adata, target_sum=1e6)
        
        # log2(x + 1)
        sc.pp.log1p(adata, base=2) 

        # restore nans
        adata.X[nan_mask] = np.nan
    return adata


# def get_exp_prof_adata(dsaids_interest, normalize:bool=False):
#     """Get Expression Profiles"""
#     # variables
#     file_dir = "/aloy/home/ddalton/projects/disease_signatures/data/DiSignAtlas/tmp/"
#     first = True
#     for dsaid in tqdm(dsaids_interest):
#         _df = pd.read_csv(os.path.join(file_dir, f"{dsaid}.csv"))
#         _adata = convert_to_adata(_df, normalize)
#         if first:
#             adata_merged = _adata
#             first = False
#         else:
#             adata_merged = ad.concat([adata_merged, _adata], join='outer', axis=0)
#     return adata_merged

def load_ncbi_gene_ids()->pd.DataFrame:
    yml_path = "/aloy/home/ddalton/projects/disease_signatures/conf/paths.yml"
    # Load YAML content
    with open(yml_path, "r") as f:
        config = yaml.safe_load(f)
    database_dir = config["database_dir"]
    return pd.read_csv(os.path.join(database_dir, "NCBI/gene_info"), sep="\t", usecols=["#tax_id", "GeneID", "type_of_gene", "Symbol"]
        )

def get_human_protein_coding_genes()->set:
    ncbi_gene_info = load_ncbi_gene_ids()
    genes = ncbi_gene_info[(ncbi_gene_info["#tax_id"] == 9606) & (ncbi_gene_info["type_of_gene"] == "protein-coding")]["Symbol"].unique()
    return sorted(list(genes))


def transpose_df_expr(df:pd.DataFrame)->pd.DataFrame:
    """
    Process the DataFrame to set gene symbols as index, transpose it, and reset index.
    """
    # set gene symbol as index
    df = df.set_index("gene_symbol")

    # transpose the DataFrame
    # this way column names will be the gene symbols and rows will the samples
    df = df.T

    # reset index so samples are no longer the index
    df = df.reset_index()

    # rename index to ID
    df = df.rename(columns={"index": "ID"})

    return df

def load_processed_data(dsaid:str)->pd.DataFrame:
    """
    Load raw data for a given DSAID.
    """
    # Path to your YAML file
    yml_path = "/aloy/home/ddalton/projects/disease_signatures/conf/paths.yml"
    # Load YAML content
    with open(yml_path, "r") as f:
        config = yaml.safe_load(f)
    database_dir = config["database_dir"]
    
    return pd.read_csv(os.path.join(database_dir, "DiSignAtlas","exp-profile-processed", f"{dsaid}.csv"))


def normalize_CPM(df:pd.DataFrame)->pd.DataFrame:
    """
    Normalize Expression data to Counts Per Million (CPM).
    """
    
    # retrieve expression - first column is gene symbols
    sample_ids = df["ID"].to_list()
    gene_cols = df.columns[1:]
    expr = df.iloc[:, 1:].astype(float).to_numpy()

    # convert dataframe to scanpy AnnData object
    adata = sc.AnnData(expr)

    # get mask of nan values
    nan_mask = np.isnan(expr)

    # replace nans with 0 for normalization
    adata.X = np.nan_to_num(adata.X, nan=0.0)

    # CPM
    sc.pp.normalize_total(adata, target_sum=1e6)
    
    # restore nans
    adata.X[nan_mask] = np.nan


    # Convert back to DataFrame
    df_normalized = pd.DataFrame(adata.X, columns=df.columns[1:], index=df.index)

    # Add ID column back
    df_normalized.insert(0, "ID", sample_ids)
    return df_normalized

def normalize_log2CPM(df:pd.DataFrame)->pd.DataFrame:
    """
    Normalize Expression data to log2 Counts Per Million (log2CPM).
    """
    # retrieve expression - first column is gene symbols
    sample_ids = df["ID"].to_list()
    gene_cols = df.columns[1:]
    expr = df.iloc[:, 1:].astype(float).to_numpy()

    # convert dataframe to scanpy AnnData object
    adata = sc.AnnData(expr)

    # get mask of nan values
    nan_mask = np.isnan(expr)

    # replace nans with 0 for normalization
    adata.X = np.nan_to_num(adata.X, nan=0.0)

    # CPM
    sc.pp.normalize_total(adata, target_sum=1e6)
    
    # log2(x + 1)
    sc.pp.log1p(adata, base=2) 

    # restore nans
    adata.X[nan_mask] = np.nan

    # Convert back to DataFrame
    df_normalized = pd.DataFrame(adata.X, columns=df.columns[1:], index=df.index)

    # Add ID column back
    df_normalized.insert(0, "ID", sample_ids)
    return df_normalized


# def get_processed_exp_prof(dsaids_interest, genes:list=None, normalize:str=None):
#     """Get Expression Profiles"""
#     # check if genes are provided if not generat
#     if genes is None:
#         genes = get_human_protein_coding_genes()

#     df_merged = pd.DataFrame(columns=["ID"] + list(genes))
#     for dsaid in tqdm(dsaids_interest):
#         try:
#             _df = load_processed_data(dsaid)
#             _df = transpose_df_expr(_df)
#             if normalize == "log2CPM":
#                 _df = normalize_log2CPM(_df)
#             elif normalize == "CPM":
#                 _df = normalize_CPM(_df)
#             df_merged = pd.concat([df_merged, _df], axis=0, join="outer", ignore_index=True)
#         except Exception as e:
#             print(f"Error processing DSAID {dsaid}: {e}")
#     return df_merged

from functools import partial
from tqdm.contrib.concurrent import process_map

def _load_one_processed(dsaid: str, normalize: str, cols):
    try:
        df = load_processed_data(dsaid)
        df = transpose_df_expr(df)
        if normalize == "log2CPM":
            df = normalize_log2CPM(df)
        elif normalize == "CPM":
            df = normalize_CPM(df)
        return df.reindex(columns=cols)
    except Exception as e:
        print(f"[warn] DSAID {dsaid}: {e}")
        return None


def get_processed_exp_prof(dsaids_interest, genes: list = None, normalize: str = None):
    if genes is None:
        genes = get_human_protein_coding_genes()
    cols = ["ID"] + list(genes)

    worker = partial(_load_one_processed, normalize=normalize, cols=cols)

    n_workers = max(1, (os.cpu_count() or 1) - 1)  # optional: leave 1 core free
    frames = process_map(worker, dsaids_interest,
                         max_workers=n_workers, chunksize=1, desc="Processing")

    frames = [f for f in frames if f is not None]
    return pd.concat(frames, axis=0, ignore_index=True, copy=False) if frames else pd.DataFrame(columns=cols)


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
        dsaid = id.split(".")[0]
        state = id.split(".")[2]
        if state == "Control":
            doid_diseases.append("Control")
            doid_ids.append("Control")
        else:
            doid_id = dsaid_2_doid.get(dsaid)
            doid_disease = doid_2_term.get(doid_id)
            
            doid_diseases.append(doid_disease)
            doid_ids.append(doid_id)

    return doid_diseases, doid_ids



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


def get_library(ids:List[str], df_info:pd.DataFrame)->List[str]:
    """Get Library
    Args:
        - ids (list): List of IDs
    Returns:
        - datasets (list): List of datasets
    """
    dsaids = [x.split(".")[0] for x in ids]
    dsaid_2_dataset = dict(zip(df_info["dsaid"], df_info["library_strategy"]))
    datasets = [str(dsaid_2_dataset[dsaid]) for dsaid in dsaids]
    return datasets

def get_tissue(ids:List[str],df_info:pd.DataFrame)->List[str]:
    """Get Tissue
    Args:
        - ids (list): List of IDs
    Returns:
        - tissues (list): List of tissues
    """
    dsaids = [x.split(".")[0] for x in ids]
    dsaid_2_tissue = dict(zip(df_info["dsaid"], df_info["tissue"]))
    tissues = [str(dsaid_2_tissue[dsaid]) for dsaid in dsaids]
    return tissues

def get_disease_study(ids:List[str],df_info:pd.DataFrame)->List[str]:
    """Get Disease Study
    Args:
        - ids (list): List of IDs
    Returns:
        - diseases (list): List of diseases
    """
    dsaids = [x.split(".")[0] for x in ids]
    dsaid_2_disease = dict(zip(df_info["dsaid"], df_info["disease"]))
    disease_study = [str(dsaid_2_disease[dsaid]) for dsaid in dsaids]
    return disease_study

def get_disease(ids:List[str],df_info:pd.DataFrame)->List[str]:
    """Get Disease
    Args:
        - ids (list): List of IDs
    Returns:
        - diseases (list): List of diseases
    """
    dsaid_2_disease = dict(zip(df_info["dsaid"], df_info["disease"]))
    diseases = list()
    for id in ids:
        dsaid = id.split(".")[0]
        state = id.split(".")[2]
        if state == "Control":
            diseases.append("Control")
        else:
            diseases.append(dsaid_2_disease.get(dsaid))
    return diseases

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

    dataset_accessions = [dsaid_2_accession[id.split(".")[0]] for id in ids]

    accession_2_id = {k: v for v, k in enumerate(set(dataset_accessions))}
    dataset_ids = [accession_2_id[accession] for accession in dataset_accessions]

    return dataset_accessions, dataset_ids

def get_dataset(ids:List[str],df_info:pd.DataFrame)->List[str]:
    """Get Dataset
    Args:
        - ids (list): List of IDs
    Returns:
        - datasets (list): List of datasets
    """
    dsaids = [x.split(".")[0] for x in ids]
    dsaid_2_dataset = dict(zip(df_info["dsaid"], df_info["accession"]))
    datasets = [str(dsaid_2_dataset[dsaid]) for dsaid in dsaids]
    return datasets

def clean_dsaids_qc(df_info:pd.DataFrame, disease_label:str="diseaseid", n_samples:int=2, n_dt:int=2):
    # count n samples per dsaid for both control and disease
    print(f"Nº dsaids: {len(df_info)}\tNº unique diseases: {df_info[disease_label].nunique()}")
    
    # filter datasets with at least n_samples in both control and disease
    df_info = df_info[(df_info['n_cases']>= 2) & (df_info['n_controls'] >= 2)]
    print(f"Filter Datasets w/ Samples +{n_samples}\tNº dsaids: {len(df_info)}\tNº unique diseases: {df_info[disease_label].nunique()}")


    # filter groups (diseases) with at least n_dt datasets
    _diseases_passed = [k for k, v in dict(df_info.groupby(disease_label)['accession'].nunique()).items() if v>=n_dt ]
    df_info = df_info[df_info[disease_label].isin(_diseases_passed)]    
    print(f"Filter Diseases w/ Datasets +{n_dt}\tNº dsaids: {len(df_info)}\tNº unique diseases: {df_info[disease_label].nunique()}")

    return df_info

def add_sample_counts(ids:str, df_info:pd.DataFrame) -> pd.DataFrame:
    dsaids = [x.split(".")[0] for x in ids]
    condition = [x.split(".")[2] for x in ids]

    # filter down df_info to only dsaids w/ info
    df_info = df_info[df_info["dsaid"].isin(dsaids)]

    # loop through dsaids and counts
    d_counts = dict()
    for i in range(len(dsaids)):
        if dsaids[i] not in d_counts:
            d_counts[dsaids[i]] = {"n_cases": 0, "n_controls": 0}
        if condition[i].lower() == "control":
            d_counts[dsaids[i]]["n_controls"] += 1
        else:
            d_counts[dsaids[i]]["n_cases"] += 1
    
    # return control and case counts in correponding order
    df_info = df_info.copy()
    df_info["n_cases"] = df_info["dsaid"].map(lambda x: d_counts[x]["n_cases"])
    df_info["n_controls"] = df_info["dsaid"].map(lambda x: d_counts[x]["n_controls"])

    return df_info


class bulk_processing:
    def __init__(self, processing:str=None, do_z_transform:bool=False, agg_genes:str=None):
        if processing is not None:
            assert processing in ["log2", "scgpt_pp", "linear"], "Err Select correct option"
        self.processing = processing
        self.do_z_transform = do_z_transform        
        self.agg_genes = agg_genes 
        self.processing = processing
        self.genes = None
        if self.agg_genes:
            assert self.agg_genes in ["keep_first", "median", "mean"], "Err Select correct option"
        
    def store_human_protein_coding_genes(self)->set:
        self.genes = self.get_human_protein_coding_genes()

    def load_ncbi_gene_ids(self)->pd.DataFrame:
        yml_path = "/aloy/home/ddalton/projects/disease_signatures/conf/paths.yml"
        # Load YAML content
        with open(yml_path, "r") as f:
            config = yaml.safe_load(f)
        database_dir = config["database_dir"]
        return pd.read_csv(os.path.join(database_dir, "NCBI/gene_info"), sep="\t", usecols=["#tax_id", "GeneID", "type_of_gene", "Symbol"]
            )

    def get_human_protein_coding_genes(self)->set:
        ncbi_gene_info = self.load_ncbi_gene_ids()
        genes = ncbi_gene_info[(ncbi_gene_info["#tax_id"] == 9606) & (ncbi_gene_info["type_of_gene"] == "protein-coding")]["Symbol"].unique()
        return sorted(list(genes))


    def _is_raw_ma(self, expr):
        """Simple Approximation to identify raw ma expr
        Rules:
            1. We have values >= 100 (not log transformed)
            2. We have no negative values (expr not centred)
        """
        # nan values can really mess this up!
        _expr = np.where(np.isnan(expr), 0, expr )
        if _expr.max() >= 100 and (_expr < 0).sum() == 0:
            return True
        return False

    def _is_raw_shifted_ma(self, expr):
        """Simple Approximation to identify raw shifted ma expr
        Rules:
            1. We have values >= 100 (not log transformed)
            2. We have negative values (expr not centred)
        """
        _expr = np.where(np.isnan(expr), 0, expr )
        if _expr.max() >= 100 and (_expr < 0).sum() > 0:
            return True
        return False

    def _is_log_ma(self, expr):
        """Simple Approximation to identify log ma expr
        Rules:
            1. We have values < 20 (log transformed)
            2. We have no negative values (expr not centred)
        """
        _expr = np.where(np.isnan(expr), 0, expr )
        if _expr.max() < 20 and (_expr < 0).sum() == 0:
            return True
        return False

    def _is_log_shifted_ma(self, expr):
        """Simple Approximation to identify log shifted ma expr
        Rules:
            1. We have values < 20 (log transformed)
            2. We have negative values (expr not centred)
        """
        _expr = np.where(np.isnan(expr), 0, expr )
        if _expr.max() < 20 and (_expr < 0).sum() > 0:
            return True
        return False


    def _log2_transform(self, expr):
        """Log transform expression expr
        """
        assert expr.min() >= 0, "expr has negative values, cannot recentre"
        # no need to handle nans - nans will stay nans 
        return np.log2(expr + 1)    # add small offset - using log2 stabilizes variance
    
    def _log1_transform(self, expr):
        """Log transform expression expr
        """
        assert expr.min() >= 0, "expr has negative values, cannot recentre"
        # no need to handle nans - nans will stay nans 
        return np.log1p(expr)    # add small offset - using log2 stabilizes variance
    

    def _recentre_expr(self, expr):
        """Recentre expr so we have no negative values
        """
        return expr - expr.min()


    def _z_transform_ma(self, expr):

        """Z transform expression expr
        """
    
        # normalize across genes for a given dataset
        # bare in mind 0 variance genes will become nans!
        # idea is that all genes contribute equally
        #! any sample with nan for a given gene will make all samples to have nans for said gene 
        #! GENE WISE!
        return scipy.stats.zscore(expr, axis=0, nan_policy='propagate')

    def _classify_expr(self, expr):
        if self._is_raw_ma(expr):
            return "raw"
        elif self._is_raw_shifted_ma(expr):
            return "raw_shifted"
        elif self._is_log_ma(expr):
            return "log"
        elif self._is_log_shifted_ma(expr):
            return "log_shifted"
        else:
            return "unknown"

    def convert_to_log2(self, df_expr):        
        # retrieve expression - first column is gene symbols
        gene_col = df_expr.columns[0]
        sample_col = df_expr.columns[1:]

        gene_ids = df_expr[gene_col].to_list()
        expr = df_expr.iloc[:, 1:].astype(float).to_numpy() # genes are rows columns are samples

        # determine expression type
        expr_type = self._classify_expr(expr)

        # process different expression types
        if expr_type == "raw":
            expr = self._log2_transform(expr)
        elif expr_type == "raw_shifted":
            expr = self._recentre_expr(expr)
            expr = self._log2_transform(expr)
        elif expr_type == "log":
            expr = expr  # already log transformed
        elif expr_type == "log_shifted":
            expr = expr 
        elif expr_type == "unknown":
            return None

        # Convert back to DataFrame
        df_expr_p = pd.DataFrame(expr, columns=sample_col, index=df_expr.index)
        
        # Add ID column back
        df_expr_p.insert(0, "gene_symbol", gene_ids)
        
        # handle duplicates
        df_expr_p = self._aggregate_genes(df_expr_p)

        return df_expr_p

    def _aggregate_genes(self, df_expr:pd.DataFrame)->pd.DataFrame:
        
        gene_col = df_expr.columns[0]  # first column is gene symbols

        if self.agg_genes == "keep_first":
            df_expr = df_expr.drop_duplicates(subset=gene_col, keep='first')
        elif self.agg_genes == "mean":
            df_expr = df_expr.groupby(gene_col, as_index=False).mean(numeric_only=True)
        elif self.agg_genes == "median":
            df_expr = df_expr.groupby(gene_col, as_index=False).median(numeric_only=True)
        
        return df_expr

    def convert_to_zscore(self, df_expr):
        # retrieve expression - first column is gene symbols
        sample_ids = df_expr["ID"].to_list()
        gene_cols = df_expr.columns[1:]
        expr = df_expr.iloc[:, 1:].astype(float).to_numpy()

        # apply z transformation GENE WISE
        #! any samples w/ nans in a gene will propagate said nans to all samples
        expr = self._z_transform_ma(expr)

        # Convert back to DataFrame
        df_expr_p = pd.DataFrame(expr, columns=gene_cols, index=df_expr.index)

        # Add ID column back
        df_expr_p.insert(0, "ID", sample_ids)
        return df_expr_p
        

    def _transpose_df_expr(self, df:pd.DataFrame)->pd.DataFrame:
        """
        Process the DataFrame to set gene symbols as index, transpose it, and reset index.
        """
        # before clean all genes w/ only nans
        df = df.dropna(how="all")
        
        # set gene symbol as index
        df = df.set_index("gene_symbol")

        # transpose the DataFrame
        # this way column names will be the gene symbols and rows will the samples
        df = df.T

        # reset index so samples are no longer the index
        df = df.reset_index()

        # rename index to ID
        df = df.rename(columns={"index": "ID"})

        return df

    def _load_processed_data(self, dsaid:str)->pd.DataFrame:
        """
        Load raw data for a given DSAID.
        """
        # Path to your YAML file
        yml_path = "/aloy/home/ddalton/projects/disease_signatures/conf/paths.yml"
        # Load YAML content
        with open(yml_path, "r") as f:
            config = yaml.safe_load(f)
        database_dir = config["database_dir"]
        
        return pd.read_csv(os.path.join(database_dir, "DiSignAtlas","exp-profile-processed", f"{dsaid}.csv"))


    def normalize_total_numpy(self, X: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
        """
        Normalize each column (cell) of a genes × cells matrix so that it sums to `target_sum`.

        Parameters
        ----------
        X : np.ndarray
            2D array (genes × cells).
        target_sum : float
            The total counts per cell after normalization.

        Returns
        -------
        X_norm : np.ndarray
            Normalized array (same shape as X).
        """
        # Sum of counts per cell, ignoring NaNs
        counts_per_cell = np.nansum(X, axis=0, keepdims=True)  # shape (1, n_cells)

        # Avoid division by zero
        counts_per_cell[counts_per_cell == 0] = 1.0

        # Normalize (NaNs in X will remain NaN)
        X_norm = X / counts_per_cell * target_sum

        return X_norm


    def convert_scgpt_pp(self, df_expr):
        
        # retrieve expression - first column is gene symbols
        gene_col = df_expr.columns[0]
        sample_col = df_expr.columns[1:]

        gene_ids = df_expr[gene_col].to_list()
        expr = df_expr.iloc[:, 1:].astype(float).to_numpy() # genes are rows columns are samples

        # determine expression type
        expr_type = self._classify_expr(expr)

        # process different expression types
        if expr_type == "raw":
            expr = self.normalize_total_numpy(expr)
            expr = self._log1_transform(expr)

        elif expr_type == "raw_shifted":
            expr = self._recentre_expr(expr)
            expr = self.normalize_total_numpy(expr)
            expr = self._log1_transform(expr)
        elif expr_type == "log":
            expr = np.power(2, expr) - 1  # revert log2
            expr = self.normalize_total_numpy(expr)
            expr = self._log1_transform(expr)
        elif expr_type == "log_shifted":
            expr = np.power(2, expr) - 1  # revert log2
            expr = self._recentre_expr(expr)
            expr = self.normalize_total_numpy(expr)
            expr = self._log1_transform(expr)
        elif expr_type == "unknown":
            return None

        # Convert back to DataFrame
        df_expr_p = pd.DataFrame(expr, columns=sample_col, index=df_expr.index)
        
        # Add ID column back
        df_expr_p.insert(0, "gene_symbol", gene_ids)
        
        # handle duplicates
        df_expr_p = self._aggregate_genes(df_expr_p)

        return df_expr_p
    

    def convert_linear(self, df_expr):
        
        # retrieve expression - first column is gene symbols
        gene_col = df_expr.columns[0]
        sample_col = df_expr.columns[1:]

        gene_ids = df_expr[gene_col].to_list()
        expr = df_expr.iloc[:, 1:].astype(float).to_numpy() # genes are rows columns are samples

        # determine expression type
        expr_type = self._classify_expr(expr)

        # process different expression types
        if expr_type == "raw":
            expr = expr # do nothing
        elif expr_type == "raw_shifted":
            expr = self._recentre_expr(expr)
        elif expr_type == "log":
            expr = np.power(2, expr) - 1  # revert log2
        elif expr_type == "log_shifted":
            expr = np.power(2, expr) - 1  # revert log2
            expr = self._recentre_expr(expr)
        elif expr_type == "unknown":
            return None

        # Convert back to DataFrame
        df_expr_p = pd.DataFrame(expr, columns=sample_col, index=df_expr.index)
        
        # Add ID column back
        df_expr_p.insert(0, "gene_symbol", gene_ids)
        
        # handle duplicates
        df_expr_p = self._aggregate_genes(df_expr_p)

        return df_expr_p
    



    def _load_one_processed(self, dsaid: str, cols):
        try:
            df = self._load_processed_data(dsaid)
            d_type = self._classify_expr(df.iloc[:, 1:].to_numpy())
            if self.processing == "log2":
                df = self.convert_to_log2(df)                   
            elif self.processing == "scgpt_pp":
                df = self.convert_scgpt_pp(df)
            elif self.processing == "linear":
                df = self.convert_linear(df)
            else:
                df = self._aggregate_genes(df)
  
            if df is None:
                print(f"DSAOID {dsaid}: Could not classify expression type {d_type}, skipping...")
                return None

            df = self._transpose_df_expr(df)                            
            if self.do_z_transform:
                df = self.convert_to_zscore(df)
            return (dsaid, d_type), df.reindex(columns=cols)
        
        except Exception as e:
            print(f"[warn] DSAID {dsaid}: {e}")
            return None


    def get_processed_exp_prof(self, dsaids_interest:list,):
        if self.genes is None:
            self.store_human_protein_coding_genes()

        # define fixed dataframe columns
        cols = ["ID"] + list(self.genes)

        # define worker function
        worker = partial(self._load_one_processed, cols=cols)
        n_workers = max(1, (os.cpu_count() or 1) - 1)  # optional: leave 1 core free

        # run parallel processing
        frames = thread_map(worker, dsaids_interest,
                            max_workers=n_workers, chunksize=1, desc="Processing")

        d_types = [f[0] for f in frames if f is not None] 
        d_types = {k:v for k,v in d_types}
        frames = [f[1] for f in frames if f is not None]

        return d_types, pd.concat(frames, axis=0, ignore_index=True, copy=False) if frames else pd.DataFrame(columns=cols)