"""Normalize GEx Data!

Normalize data using scGPT preprocessor. Also filter genes used for normalization.
"""
# 1. Imports, Variables, Functions
# imports
from scgpt.preprocess import Preprocessor
import numpy as np
import logging
import scanpy as sc
import anndata as ad
import anndata as AnnData
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# variables
data_path = ""
data_is_raw = True
filter_gene_by_counts = False
n_bins=51
gene_filtering = "top_presence"
max_seq_len=3501


# functions
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

# 2. Load Data
# load data
adata = sc.read(data_path)


# filter genes
if gene_filtering == "top_presence":
    mask_genes = get_top_k_most_present_genes(adata, 
                    k=max_seq_len)


logging.info(f"Combined mask {np.sum(mask_genes)} genes left")

# mask the genes
adata = adata[:, mask_genes]




# preprocess
preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=filter_gene_by_counts,  # step 1
    filter_cell_by_counts=False,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=data_is_raw,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)

preprocessor(adata, batch_key=None)


# save
sc.write(data_path, adata)