
import logging
import numpy as np
import os
import pickle
from tqdm import tqdm
from anndata import AnnData
import pandas as pd
from typing import *
from sklearn.model_selection import StratifiedKFold, KFold
import torch, psutil, wandb, random
from scanpy.pp import combat


def get_test_split_common(obs: pd.DataFrame, n_splits=5) -> List[str]:
    """Get Test Split
    We will perform a split for those diseases which have more than one dataset.

    Ther MUST not be any data-leakage between the train and test set - no shared datasets between the two sets.

    Strategy:
        1. Check diseases w/ 5+ datasets
        2. Divide dataset into train and test w/ 4:1 ratio
        3. Assign train and test to the respective datasets

    """


    obs_copy = obs.copy(deep=True)

    combined_labels = (
        obs_copy["celltype"].astype(str) + "_" + obs_copy["dataset_id"].astype(str)
    )

    combined_labels.unique()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    for i, (train_idx, test_idx) in enumerate(
        skf.split(X=obs_copy["ids"], y=combined_labels)
    ):

        mask = np.zeros(len(obs_copy), dtype=bool)
        mask[test_idx] = True

        # 3. Assign train and test labels
        obs_copy[f"test_split_{i+1}"] = mask

        logging.info(
            f"Nº of diseases in train split {i+1}: {obs_copy.iloc[train_idx]['celltype'].nunique()}"
        )

        logging.info(
            f"Nº of diseases in test split {i+1}: {obs_copy.iloc[test_idx]['celltype'].nunique()}"
        )

        logging.info(
            f"Nº of datasets in train split {i+1}: {obs_copy.iloc[train_idx]['dataset_id'].nunique()}"
        )

        logging.info(
            f"Nº of datasets in test split {i+1}: {obs_copy.iloc[test_idx]['dataset_id'].nunique()}"
        )

        logging.info(
            f"Nº of samples in train split {i+1}: {obs_copy.iloc[train_idx]['ids'].nunique()}"
        )

        logging.info(
            f"Nº of samples in test split {i+1}: {obs_copy.iloc[test_idx]['ids'].nunique()}"
        )

    return obs_copy

def get_top_k_most_present_genes(
    adata: AnnData,
    k: int = 3501,
) -> np.array:
    """
    Filters genes based on the top `k` most present genes across samples.
    If there are more genes with the same presence, the top `k` will be selected
    based on variance as a secondary criterion.

    Parameters:
    - adata: AnnData object containing gene expression data
    - k: The number of top genes to retain based on their presence across samples

    Returns:
    - A boolean array (mask) indicating which genes to keep.
    """
    total_n_genes = adata.X.shape[1]
    total_n_samples = adata.X.shape[0]

    logging.info(f"Nº genes: {total_n_genes}, Nº samples: {total_n_samples}")

    # Step 1: Count gene presence (non-NaN and non-zero values) across samples
    count_presence = np.sum(~np.isnan(adata.X), axis=0)

    # Step 2: Sort genes by presence in descending order
    sorted_indices = np.argsort(count_presence)[::-1]

    # Step 3: Handle the case where multiple genes have the same presence
    top_k_indices = sorted_indices[:k]
    
    if k < len(sorted_indices) and count_presence[top_k_indices[-1]] == count_presence[sorted_indices[k]]:
        min_presence = count_presence[top_k_indices[-1]]

        # Step 4: Find all genes with the same presence as the cutoff
        equal_presence_indices = sorted_indices[count_presence[sorted_indices] == min_presence]

        # Step 5: If the number of equal presence genes exceeds `k`, use variance as a tie-breaker
        remaining_slots = k - np.sum(count_presence[sorted_indices[:k]] > min_presence)
        if len(equal_presence_indices) > remaining_slots:
            # Compute variance for all genes
            gene_variance = np.nanstd(adata.X, axis=0)
            
            # substitute nan values with -infinite values
            gene_variance = np.nan_to_num(gene_variance, nan=-np.inf)
            
            # Sort equal presence genes by variance in descending order
            sorted_by_variance = equal_presence_indices[np.argsort(gene_variance[equal_presence_indices])[::-1]]

            # Replace the last entries in the top `k` with highest variance genes from the tie
            top_k_indices = np.concatenate(
                [sorted_indices[count_presence[sorted_indices] > min_presence], 
                 sorted_by_variance[:remaining_slots]]
            )
        else:
            top_k_indices = sorted_indices[:k]

    # Step 6: Create a boolean mask for the top `k` genes
    mask_top_k = np.zeros(total_n_genes, dtype=bool)
    mask_top_k[top_k_indices] = True

    logging.info(f"Top {k} most present genes selected. - >= {min(count_presence[mask_top_k])}")

    return mask_top_k


def get_top_k_highest_variance_genes(
    adata: AnnData,
    k: int = 3501,
) -> np.array:
    """
    Filters genes based on the top `k` genes with the highest variance across samples.

    Parameters:
    - adata: AnnData object containing gene expression data
    - k: The number of top genes to retain based on their variance across samples

    Returns:
    - A boolean array (mask) indicating which genes to keep.
    """
    total_n_genes = adata.X.shape[1]
    total_n_samples = adata.X.shape[0]

    logging.info(f"Nº genes: {total_n_genes}, Nº samples: {total_n_samples}")

    # Step 1: Calculate variance for each gene
    gene_variance = np.nanstd(adata.X, axis=0)
    
    # substitute nan values with -infinite values
    gene_variance = np.nan_to_num(gene_variance, nan=-np.inf)

    # Step 2: Sort genes by variance in descending order
    sorted_indices_by_variance = np.argsort(gene_variance)[::-1]

    # Step 3: Select the top `k` genes with the highest variance
    top_k_indices = sorted_indices_by_variance[:k]

    # Step 4: Create a boolean mask for the top `k` genes
    mask_top_k = np.zeros(total_n_genes, dtype=bool)
    mask_top_k[top_k_indices] = True

    logging.info(f"Top {k} highest variance genes selected.")

    return mask_top_k

def get_k_random_genes(
    adata: AnnData,
    k: int = 3501,
    random_seed: int = None
) -> np.array:
    """
    Randomly selects `k` genes from the gene expression data.

    Parameters:
    - adata: AnnData object containing gene expression data
    - k: The number of random genes to select
    - random_seed: An optional random seed for reproducibility

    Returns:
    - A boolean array (mask) indicating which genes were randomly selected.
    """
    total_n_genes = adata.X.shape[1]

    # Ensure that k is not larger than the total number of genes
    if k > total_n_genes:
        raise ValueError(f"k cannot be larger than the total number of genes ({total_n_genes}).")

    logging.info(f"Nº genes: {total_n_genes}, selecting {k} random genes")

    # Set the random seed for reproducibility (if provided)
    if random_seed is not None:
        np.random.seed(random_seed)

    # Step 1: Randomly select `k` indices from the total number of genes
    random_indices = np.random.choice(total_n_genes, size=k, replace=False)

    # Step 2: Create a boolean mask for the selected genes
    mask_random_k = np.zeros(total_n_genes, dtype=bool)
    mask_random_k[random_indices] = True

    logging.info(f"{k} random genes selected.")

    return mask_random_k

def get_top_k_highest_variance_genes_with_presence(
    adata: AnnData,
    k: int = 3501,
    presence_pct: float = 0.8
) -> np.array:
    """
    Filters genes based on the top `k` genes with the highest variance across samples,
    but only includes genes that have presence (non-NaN, non-zero values) in more than
    a specified percentage of samples.

    Parameters:
    - adata: AnnData object containing gene expression data
    - k: The number of top genes to retain based on their variance across samples
    - presence_pct: The minimum percentage of samples a gene must be present in to be considered

    Returns:
    - A boolean array (mask) indicating which genes to keep.
    """
    total_n_genes = adata.X.shape[1]
    total_n_samples = adata.X.shape[0]
    
    logging.info(f"Nº genes: {total_n_genes}, Nº samples: {total_n_samples}")

    # Step 1: Calculate gene presence (non-NaN, non-zero values) across samples
    count_presence = np.sum(~np.isnan(adata.X) & (adata.X != 0), axis=0)

    # Step 2: Calculate the presence percentage for each gene
    presence_threshold = total_n_samples * presence_pct
    valid_genes_mask = count_presence >= presence_threshold

    logging.info(f"Presence threshold set at {presence_threshold} samples (presence_pct = {presence_pct * 100}%)")
    logging.info(f"{np.sum(valid_genes_mask)} genes meet the presence criterion")

    # Step 3: Calculate variance for genes that meet the presence threshold
    gene_variance = np.nanstd(adata.X, axis=0)

    # substitute nan values with -infinite values
    gene_variance = np.nan_to_num(gene_variance, nan=-np.inf)

    # Step 4: Select the indices of genes that meet the presence threshold
    valid_genes_variance = gene_variance[valid_genes_mask]

    # Step 5: Sort the valid genes by variance in descending order
    sorted_indices_by_variance = np.argsort(valid_genes_variance)[::-1]

    # Step 6: Select the top `k` genes with the highest variance that meet the presence criterion
    top_k_indices = np.where(valid_genes_mask)[0][sorted_indices_by_variance[:k]]

    # Step 7: Create a boolean mask for the top `k` genes
    mask_top_k = np.zeros(total_n_genes, dtype=bool)
    mask_top_k[top_k_indices] = True

    logging.info(f"Top {k} highest variance genes with presence above {presence_pct * 100}% selected.")

    return mask_top_k

def get_test_split(obs: pd.DataFrame, n_splits=5) -> List[str]:
    """Get Test Split
    We will perform a split for those diseases which have more than one dataset.

    Ther MUST not be any data-leakage between the train and test set - no shared datasets between the two sets.

    Strategy:
        1. Check diseases w/ 5+ datasets
        2. Divide dataset into train and test w/ 4:1 ratio
        3. Assign train and test to the respective datasets

    """

    obs_copy = obs.copy(deep=True)

    all_diseases = obs_copy["celltype"].unique()

    logging.info(f"Number of diseases: {len(all_diseases)}")

    # 2. Divide dataset into train and test w/ 4:1 ratio
    kf = KFold(n_splits=n_splits, shuffle=True)
    for i, (train_idx, test_idx) in enumerate(
        kf.split(
            X=obs_copy["ids"],
            y=obs_copy["celltype"],
        )
    ):

        mask = np.zeros(len(obs_copy), dtype=bool)
        mask[test_idx] = True

        # 3. Assign train and test labels
        obs_copy[f"test_split_{i+1}"] = mask

        logging.info(
            f"Nº of diseases in train split {i+1}: {obs_copy.iloc[train_idx]['celltype'].nunique()}"
        )

        logging.info(
            f"Nº of diseases in test split {i+1}: {obs_copy.iloc[test_idx]['celltype'].nunique()}"
        )

        logging.info(
            f"Nº of datasets in train split {i+1}: {obs_copy.iloc[train_idx]['dataset_id'].nunique()}"
        )

        logging.info(
            f"Nº of datasets in test split {i+1}: {obs_copy.iloc[test_idx]['dataset_id'].nunique()}"
        )

        logging.info(
            f"Nº of samples in train split {i+1}: {obs_copy.iloc[train_idx]['ids'].nunique()}"
        )

        logging.info(
            f"Nº of samples in test split {i+1}: {obs_copy.iloc[test_idx]['ids'].nunique()}"
        )

    return obs_copy


def split_stratified(
    df: pd.DataFrame,
    y_label: str = "celltype",
    group_label: str = "dataset_id",
    split_size: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Get a single stratified test split for the provided observations.
    We are here splitting datasets - trying to ensure for each label(disease) we hav at least one dataset in both train and test.
    """
    random.seed(seed)

    df = df.copy(deep=True)

    # (tiny guard) each label must span ≥2 datasets to appear in both splits
    ds_per_label = df.groupby(y_label)[group_label].nunique()
    if (ds_per_label < 2).any():
        raise ValueError("Some labels occur in <2 datasets; cannot place them in both splits.")

    train_groups = set()
    test_groups  = set()       #

    # loop through labels (sorted for determinism; remove 'sorted' if you prefer)
    for y_i in sorted(df[y_label].unique()):
        _df_y = df[df[y_label] == y_i]

        # groups for this label
        _groups = _df_y[group_label].unique().tolist()

        # exclude anything already fixed to either side
        already_train     = set(_groups) & train_groups
        already_test      = set(_groups) & test_groups
        _groups_to_split  = list(set(_groups) - already_train - already_test)

        if len(_groups_to_split) == 0:
            # nothing left to place for this label (already covered)
            print(f"Skipping {y_i} as already placed everything in train/test.")
            continue

        if len(_groups_to_split) == 1:
            g = _groups_to_split[0]
            if not already_test:
                # ensure the label appears in TEST at least once
                test_groups.add(g)
            elif not already_train:
                # ensure the label appears in TRAIN at least once
                train_groups.add(g)
            else:
                # both sides already have this label; keep your original choice
                train_groups.add(g)

        _s_size = max(1, len(_groups_to_split) // split_size)  # at least 1 to test
        # if nothing from this label is in train yet, don't send them all to test
        if not already_train and _s_size >= len(_groups_to_split):
            _s_size = len(_groups_to_split) - 1  # leave >=1 for train

        # pick test groups and record both sides
        _test_groups = set(random.sample(_groups_to_split, _s_size))
        test_groups.update(_test_groups)                            
        train_groups.update(set(_groups_to_split) - _test_groups)

    # final assignment STRICTLY by dataset_id membership in test_groups
    df["test_split_1"] = df[group_label].isin(test_groups).astype(int)
    return df

def is_test_good(df_split_2, group_label:str="dataset_id", i=1):
    # check if there is no data leaking 
    # must NOT be any datasets in both splits
    test_dt = df_split_2[df_split_2[f"test_split_{i}"]==1][group_label].unique()
    train_dt = df_split_2[df_split_2[f"test_split_{i}"]==0][group_label].unique()

    return len(set(test_dt) & set(train_dt)) == 0
def log_cpu_memory_usage():
    memory = psutil.virtual_memory()
    logging.info(f"CPU Memory Usage: {memory.percent}% used of {memory.total / (1024 ** 3):.2f} GB total")


def log_gpu_memory_usage():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            cached = torch.cuda.memory_reserved(i) / (1024 ** 3)
            logging.info(f"GPU {i} Memory Usage: {allocated:.2f} GB allocated, {cached:.2f} GB cached")



def get_pos_weight(y:np.array, max_cap=50.0, min_cap=1.0) -> torch.Tensor:
    """
    Compute positive weights for BCEWithLogitsLoss based on class imbalance.
    
    Args:
        y (torch.Tensor): Binary labels of shape (N, num_classes).
        max_cap (float): Maximum cap for the weights to avoid extreme imbalance.
        
    Returns:
        torch.Tensor: Positive weights for each class.
    """
    # convert numpy to torch tensor if needed
    if isinstance(y, np.ndarray):
        y = torch.tensor(y, dtype=torch.float32)

    # Count positives and negatives per label
    pos_counts = y.sum(dim=0)                  # (#labels,)
    neg_counts = y.shape[0] - pos_counts       # (#labels,)

    # Compute pos_weight = negatives / positives
    pos_weight = neg_counts / pos_counts.clamp(min=1)  # avoid division by zero



    # Cap the values to avoid extreme imbalance exploding gradients
    pos_weight = pos_weight.clamp(max=max_cap, min=min_cap)  

    return pos_weight

def perform_combat_correction(adata) -> AnnData:
    """
    Perform ComBat batch correction on the input AnnData object.

    Parameters:
    - adata: AnnData
        The AnnData object containing the expression data and batch information.

    Returns:
    - AnnData
        The AnnData object with batch-corrected expression data.
    """
    logging.info("Performing batch correction using ComBat.")

    # Impute NaN values with 0 for ComBat compatibility
    imputed_X = adata.X.copy()
    mask_nan = np.isnan(imputed_X)  # Identify original NaN positions
    imputed_X[mask_nan] = 0

    # Temporarily replace `adata.X` with imputed values
    adata.X = imputed_X

    # Apply ComBat for batch correction
    combat(adata, key="batch_id")

    # Restore original NaN values in the batch-corrected matrix
    adata.X[mask_nan] = np.nan

    logging.info("Batch correction completed.")

    return adata


def define_wandb_metrcis():
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")


def report_split(df:pd.DataFrame, disease_label:str="celltype", split_idx:int = 1)->None:
    
    # filter
    df_train = df[df[f"test_split_{split_idx}"]==0]
    df_test = df[df[f"test_split_{split_idx}"]==1]
    
    # report
    print(f"Nº of diseases in train split {split_idx}:\t{df_train[disease_label].nunique()}")
    print(f"Nº of diseases in test split {split_idx}:\t{df_test[disease_label].nunique()}")
    print(f"Nº of datasets in train split {split_idx}:\t{df_train['dataset_id'].nunique()}")
    print(f"Nº of datasets in test split {split_idx}:\t{df_test['dataset_id'].nunique()}")
    print(f"Nº of samples in train split {split_idx}:\t{df_train['ids'].nunique()}")
    print(f"Nº of samples in test split {split_idx}:\t{df_test['ids'].nunique()}")