from typing import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import itertools
import pandas as pd
def plot_kde(X:List, labels:List, colors:List, title:str, metric:str = "Cosine", weights:List=None)->None:
    """Plot KDE"""

    # Plot
    plt.figure(figsize=(4, 4), dpi=300)

    for i in range(len(X)):
        # background distribution
        if "random" in labels[i].lower() or "unrelated" in labels[i].lower():
            sns.kdeplot(
                x=X[i],
                linewidth=3,
                linestyle="--",
                label=labels[i],
                fill=True,
                color=colors[i],
                zorder=2,
                
            )

        else:
            sns.kdeplot(
                x=X[i],
                linewidth=3,
                label=labels[i],
                fill=True,
                color=colors[i],
                zorder=3 if i < 2 else 2,
                weights=weights[i] if weights is not None else None,
                )
    plt.title("Cosine Similarity: Same Disease vs Random")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.xlim(0, 1)
    plt.grid(zorder=-3, linestyle="--")
    plt.show()


def get_same_disease_sim(adata_obs:pd.DataFrame, s_matrix:np.array)->Tuple[Tuple[np.array, np.array],Tuple[np.array, np.array]]:
    """Get same disease similarity"""
    
    # add control column
    adata_obs = adata_obs.copy()
    adata_obs["is_control"] = ["C" if i.lower().startswith("control") else "D" for i in adata_obs["do_id_study"]]

    # Precompute DOID-to-sample indices
    adata_obs.reset_index(drop=True, inplace=True)  # in case reset index 
    doid_to_indices = dict()
    for idx, doid in adata_obs["do_id_study"].items():
        if doid not in doid_to_indices:
            doid_to_indices[doid] = []
        doid_to_indices[doid].append(idx)


    # Same Disease Same Dataset
    c_same_same, w_same_same = [], []

    # Same Disease Diff Dataset
    c_same_diff, w_same_diff = [], []


    # Process same disease pairs
    for do_id in tqdm(adata_obs["do_id_study"].unique(), desc="Processing same disease pairs"): # loop through all unique diseases
        if do_id.startswith("Control"): # skip controls
            continue

        # loop through unique datasets and add same disease same dataset pairs
        _df = adata_obs.query("do_id_study == @do_id") 
        datasets = _df["dataset"].unique()
        for dt in datasets:
            
            # retrieve same disease same dataset pairs
            idxs = _df.query("dataset == @dt").index
            if len(idxs) < 2:
                continue

            # get all combinations of pairs
            idxs_p = np.array(list(itertools.combinations(idxs, 2)))
            c_same_same.extend(s_matrix[idxs_p[:, 0], idxs_p[:, 1]])
            w_same_same.extend([len(idxs_p)] * len(idxs_p))

        # if there are multiple dataset we can compute same disease diff dataset pairs
        if len(datasets) > 1:

            # loop through all combinations of datasets
            for d1, d2 in itertools.combinations(datasets, 2):

                idxs1 = _df.query("dataset == @d1").index
                idxs2 = _df.query("dataset == @d2").index
    
                # all combinations of pairs
                idxs_p = np.array(list(itertools.product(idxs1, idxs2)))
                c_same_diff.extend(s_matrix[idxs_p[:, 0], idxs_p[:, 1]])
                w_same_diff.extend([len(idxs_p)] * len(idxs_p))

    # Normalize weights
    w_same_same = np.array(w_same_same) / len(c_same_same)
    w_same_diff = np.array(w_same_diff) / len(c_same_diff)

    # convert to numpy arrays
    c_same_same = np.array(c_same_same)
    c_same_diff = np.array(c_same_diff)

    return (c_same_same, w_same_same), (c_same_diff, w_same_diff)


def get_unrelated_pairs(
    adata_obs: pd.DataFrame,
    df_related: pd.DataFrame,
    s_matrix: np.ndarray,
    n_samples: int = 1000,
    seed: int = 42,
) -> List[Tuple[int, int]]:
    """Get unrelated pairs of samples"""

    np.random.seed(seed)

    adata_obs = adata_obs.copy()
    adata_obs.reset_index(drop=True, inplace=True)
    adata_obs["is_control"] = ["C" if i.startswith("Control") else "D" for i in adata_obs["do_id_study"]]

    # Precompute DOID-to-sample indices
    doid_to_indices = dict()
    for idx, doid in adata_obs["do_id_study"].items():
        if doid not in doid_to_indices:
            doid_to_indices[doid] = []
        doid_to_indices[doid].append(idx)

    # Build unrelated pairs
    unrelated_pairs = []
    pair_key_set = set(tuple(sorted(p)) for p in df_related["pair_sorted"])
    doids = list(doid_to_indices.keys())
    print(len(doids), "diseases in dataset")
    for d1, d2 in tqdm(itertools.combinations(doids, 2)):

        # skip related diseases
        if d1 == d2 or tuple(sorted([d1, d2])) in pair_key_set:
            continue    
        
        #! WARNING - THIS WOULD ALSO INCLUDE CONTROLS
        # idxs1 = adata_obs.query("do_id_study == @d1").index
        # idxs2 = adata_obs.query("do_id_study == @d2").index
        idxs1 = adata_obs.query("do_id_study == @d1").index
        idxs2 = adata_obs.query("do_id_study == @d2").index

        unrelated_pairs.extend(itertools.product(idxs1, idxs2))


    # Sample unrelated pairs
    print(len(unrelated_pairs), "unrelated pairs found")
    k = min(n_samples, len(unrelated_pairs))
    print(k)
    selected_idxs = np.random.choice(len(unrelated_pairs), size=k, replace=False)
    print("Selected", len(selected_idxs), "unrelated pairs")
    _rand_sample_pairs = [unrelated_pairs[i] for i in selected_idxs]
    print("Sampled pairs:", len(_rand_sample_pairs))
    c_rand = [s_matrix[i, j] for i, j in _rand_sample_pairs]
    
    
    return c_rand

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