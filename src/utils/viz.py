from typing import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import itertools
import pandas as pd
from collections import Counter
import sys
sys.path.append("../../")
from src.utils import utils as ut
import matplotlib.pyplot as plt
import random

def plot_kde(X:List, labels:List, colors:List, title:str, metric:str = "Cosine", weights:List=None, sample:int=np.inf)->None:
    """Plot KDE"""

    # Plot
    plt.figure(figsize=(4, 4), dpi=300)



    for i in range(len(X)):
        X_i = X[i]
        w_i = weights[i] if weights is not None else None
        
        if len(X_i) > sample:
            _sample_i = random.sample(range(len(X_i)), sample)
            X_i = X_i[_sample_i]
            if weights is not None:
                w_i = w_i[_sample_i]

        # background distribution
        if "random" in labels[i].lower() or "unrelated" in labels[i].lower():
            sns.kdeplot(
                x=X_i,
                linewidth=3,
                linestyle="--",
                label=labels[i],
                fill=True,
                color=colors[i],
                zorder=2,
                
            )

        else:
            sns.kdeplot(
                x=X_i,
                linewidth=3,
                label=labels[i],
                fill=True,
                color=colors[i],
                zorder=3 if i < 2 else 2,
                weights=w_i if weights is not None else None,
                )
    plt.title(title)
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
    adata_obs["sample_type"] = ["C" if i.lower().startswith("control") else "D" for i in adata_obs["doid_id"]]

    # Precompute DOID-to-sample indices
    adata_obs.reset_index(drop=True, inplace=True)  # in case reset index 
    # Same Disease Same Dataset
    c_same_same, l_same_same = [], []

    # Same Disease Diff Dataset
    c_same_diff, l_same_diff = [], []

    # Control Same Dataset
    c_ctrl_same, l_ctrl_same = [], []

    # Control Diff Dataset
    c_ctrl_diff, l_ctrl_diff = [], []

    # Process same disease pairs
    for do_id in tqdm(adata_obs["doid_id"].unique(), desc="Processing same disease pairs"): # loop through all unique diseases
        if do_id.startswith("Control"): # skip controls
            continue

        # loop through unique datasets and add same disease same dataset pairs
        datasets = adata_obs.query("doid_id == @do_id")["dataset"].unique() 
        _df = adata_obs.query("dataset in @datasets")   # dataframe w/ datasets from current disease
        for dt in datasets:
            
            # retrieve same disease same dataset pairs
            idxs = _df.query("(dataset == @dt) & (doid_id == @do_id)").index
            if len(idxs) < 2:
                continue

            # get all combinations of pairs
            idxs_p = np.array(list(itertools.combinations(idxs, 2)))
            c_same_same.extend(s_matrix[idxs_p[:, 0], idxs_p[:, 1]])
            l_same_same.extend([do_id] * len(idxs_p))

            # retrieve control same dataset pairs
            idxs_c = _df.query("(dataset == @dt) & (sample_type == 'C')").index
            if len(idxs_c) < 1:
                continue

            # get all combinations of pairs
            idxs_p = np.array(list(itertools.product(idxs, idxs_c)))
            c_ctrl_same.extend(s_matrix[idxs_p[:, 0], idxs_p[:, 1]])
            l_ctrl_same.extend([do_id] * len(idxs_p))

        # if there are multiple dataset we can compute same disease diff dataset pairs
        if len(datasets) > 1:

            # loop through all combinations of datasets
            for d1, d2 in itertools.combinations(datasets, 2):
                
                idxs1 = _df.query("(dataset == @d1) & (doid_id == @do_id)").index
                idxs2 = _df.query("(dataset == @d2) & (doid_id == @do_id)").index
                idxs3 = _df.query("(dataset == @d2) & (sample_type == 'C')").index    #! possibly use dsaid rather than dataset?
                # all combinations of pairs
                
                #! CORRECT THE WEIGHTS
                # get all combinations of same disease diff dataset pairs
                idxs_p = np.array(list(itertools.product(idxs1, idxs2)))
                if len(idxs_p) > 0:
                    c_same_diff.extend(s_matrix[idxs_p[:, 0], idxs_p[:, 1]])
                    l_same_diff.extend([do_id] * len(idxs_p))


                # get all combinations of control diff dataset pairs
                idxs_p = np.array(list(itertools.product(idxs1, idxs3)))
                if len(idxs_p) > 0:
                    c_ctrl_diff.extend(s_matrix[idxs_p[:, 0], idxs_p[:, 1]])
                    l_ctrl_diff.extend([do_id] * len(idxs_p))

    def _convert_labels_to_weights(labels: List[str]) -> List[float]:
        """Convert labels to weights"""
        counts = Counter(labels)
        w = np.array([1.0 / counts[d] for d in labels], dtype=float)
        w /= w.sum()                               # normalize to sum to 1
        return w

    # convert labels to weights
    l_same_same = _convert_labels_to_weights(l_same_same)
    l_same_diff = _convert_labels_to_weights(l_same_diff)
    l_ctrl_same = _convert_labels_to_weights(l_ctrl_same)
    l_ctrl_diff = _convert_labels_to_weights(l_ctrl_diff)

    # convert to numpy arrays
    c_same_same = np.array(c_same_same)
    c_same_diff = np.array(c_same_diff)
    c_ctrl_same = np.array(c_ctrl_same)
    c_ctrl_diff = np.array(c_ctrl_diff)

    return (c_same_same, l_same_same), (c_same_diff, l_same_diff), (c_ctrl_same, l_ctrl_same), (c_ctrl_diff, l_ctrl_diff)


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
    adata_obs["is_control"] = ["C" if i.startswith("Control") else "D" for i in adata_obs["doid_id"]]

    # Precompute DOID-to-sample indices
    doid_to_indices = dict()
    for idx, doid in adata_obs["doid_id"].items():
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
        # idxs1 = adata_obs.query("doid_id == @d1").index
        # idxs2 = adata_obs.query("doid_id == @d2").index
        idxs1 = adata_obs.query("doid_id == @d1").index
        idxs2 = adata_obs.query("doid_id == @d2").index

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


def plot_adata_top_families(adata):
    # get Disease Ontology graph
    do_g = ut.load_do_graph()

    # get nodes
    _class_nodes = ut.get_lvl1_nodes(do_g)

    # Generate multilabel vectors for level 1 nodes
    Y_multilabel, _class_nodes = ut.generate_multilabel_vectors(
        adata, do_g, _class_nodes
    )
    print(
        f"Generated multilabel vectors for level 1 nodes with shape {Y_multilabel.shape}"
    )

    # Clean multilabel vectors by removing nodes with no samples
    Y_multilabel, _class_nodes = ut.clean_multilabel_vectors(Y_multilabel, _class_nodes)
    print(
        f"Cleaned multilabel vectors for top 50 nodes with shape {Y_multilabel.shape}"
    )

    # Check the multilabel vector for level 1 nodes
    ut.check_multilabel_vector(Y_multilabel, _class_nodes, do_g)

    # get doids and class names
    _, Y_multilabel_name = ut.get_multilabel_data(
        Y_multilabel, _class_nodes, do_g
    ) 

    flatten = lambda x: [i for sublist in x for i in sublist]

    _all_dis = flatten(Y_multilabel_name)
    counts_dis = Counter(_all_dis)

    # clean
    counts_dis = {k:v for k, v in counts_dis.items() if (v > 0)& (k.lower() != "control")}

    # sort in descending order
    counts_dis = dict(sorted(counts_dis.items(), key=lambda item: item[1], reverse=False))

    # plot horizontal histogram of counts of main families
    plt.figure(figsize=(4, 6), dpi=100)
    plt.barh(list(counts_dis.keys()), list(counts_dis.values()))
    plt.xlabel("Number of Samples")
    plt.ylabel("Disease Families")
    plt.title("Nº Samples per Disease Family")
    plt.show()

