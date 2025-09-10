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
from sklearn.metrics import roc_curve, auc
from matplotlib.ticker import EngFormatter
import scanpy as sc
from sklearn.manifold import TSNE
from matplotlib import cm

def plot_kde(X:List, labels:List, colors:List, title:str, metric:str = "Cosine", weights:List=None, sample:int=np.inf, dpi:int=300, output_dir:str=None)->None:
    """Plot KDE"""
    # Plot
    plt.figure(figsize=(4, 4), dpi=dpi)



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
    # plt.legend()
    plt.tight_layout()
    plt.xlim(0, 1)
    plt.grid(zorder=-3, linestyle="--")
    
    # legend centered below the axes
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20))

    # make room for the legend
    plt.subplots_adjust(bottom=0.25)

    if output_dir:
        plt.savefig(output_dir, dpi=dpi, bbox_inches='tight')
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
    w_same_same = _convert_labels_to_weights(l_same_same)
    w_same_diff = _convert_labels_to_weights(l_same_diff)
    w_ctrl_same = _convert_labels_to_weights(l_ctrl_same)
    w_ctrl_diff = _convert_labels_to_weights(l_ctrl_diff)

    # convert to numpy arrays
    c_same_same = np.array(c_same_same)
    c_same_diff = np.array(c_same_diff)
    c_ctrl_same = np.array(c_ctrl_same)
    c_ctrl_diff = np.array(c_ctrl_diff)

    return (c_same_same,c_same_diff,c_ctrl_same, c_ctrl_diff), (w_same_same, w_same_diff, w_ctrl_same, w_ctrl_diff)




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


def plot_adata_top_families(adata, n_class_nodes:int=None, top_n:int=None, dpi:int=150)->None:
    # get Disease Ontology graph
    do_g = ut.load_do_graph()

    # get nodes
    if n_class_nodes:
        # get Disease Ontology graph
        do_g = ut.load_do_graph()

        # get sanchez IC
        doid_2_ic = ut.get_sanchez_ic(do_g)
        _class_nodes = ut.get_n_lowest_ic_nodes(doid_2_ic, n_class_nodes)

    else:
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
    plt.figure(figsize=(3, 3), dpi=dpi)
    if top_n:
        plt.barh(list(counts_dis.keys())[-top_n:], list(counts_dis.values())[-top_n:],zorder=2)
        plt.title(f"Nº Samples per Top {top_n} Disease Family")
    else:
        plt.barh(list(counts_dis.keys()), list(counts_dis.values()),zorder=2)
        plt.title("Nº Samples per Disease Family")
    plt.xlabel("Number of Samples")
    plt.ylabel("Disease Families")
    plt.grid(linestyle="--", zorder=-1, alpha=0.5, axis="x")
    plt.show()


def get_roc(y_pos_score:List, y_neg_score:List, weights=None, sample=np.inf)->float:

    if len(y_neg_score) > sample:
        _idxs = random.sample(range(len(y_neg_score)), sample)
        y_neg_score = y_neg_score[_idxs]


    if len(y_pos_score) > sample:
        _idxs = random.sample(range(len(y_pos_score)), sample)
        y_pos_score = y_pos_score[_idxs]
        weights = weights[_idxs] if (weights is not None) else None

    print(f"Nº of positive disease pairs: {len(y_pos_score)}")
    
    # ROC computation
    y_true = np.concatenate([np.ones(len(y_pos_score)), np.zeros(len(y_neg_score))])    # construct y true based on negatives and positives
    scores = np.concatenate([y_pos_score, y_neg_score])
    weights = np.concatenate([weights, np.ones(len(y_neg_score))]) if weights is not None else None

    fpr, tpr, _ = roc_curve(y_true, scores, sample_weight=weights)
    roc_auc = auc(fpr, tpr)
    return roc_auc, fpr, tpr

def plot_roc(
    Y_pos:List[np.array], y_neg_score:np.array, colors:List[str], labels:List[str], title:str="AUROC",w_pos:List[np.array]=None, sample:int=np.inf, dpi:int=300, output_dir:str=None
):

    # process the labels
    labels = [x.split("|")[0] for x in labels]

    plt.figure(figsize=(4, 4), dpi=dpi)

    for i in range(len(Y_pos)):
        Y_pos_i = Y_pos[i]
        # when slicing per-disease
        w_pos_i = w_pos[i] if (w_pos is not None) else None

        
        print(f"Nº of positive disease pairs: {len(Y_pos_i)}")
        
        roc_auc, fpr, tpr = get_roc(Y_pos_i, y_neg_score, weights=w_pos_i, sample=sample)
        plt.plot(
            fpr,
            tpr,
            label=labels[i] + f" (AUROC = {roc_auc:.2f})",
            color=colors[i],
            linewidth=3,
        )
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.grid(linestyle="--", zorder=-1)
    
    if output_dir:
        plt.savefig(output_dir, dpi=dpi,  bbox_inches='tight')
    plt.show()

def _convert_labels_to_weights(labels: List[str]) -> List[float]:
    """Convert labels to weights"""
    counts = Counter(labels)
    w = np.array([1.0 / counts[d] for d in labels], dtype=float)
    w /= w.sum()                               # normalize to sum to 1
    return w

def get_same_dis_similarity_splits(adata_qry, adata_ref,s_matrix,dsaid_2_dis)->Tuple[Tuple,Tuple]:
    """
    Get same disease similarity across different splits (ie test vs train)
    """

    # get list of diseases
    uniq_do_ids = list(adata_qry.obs["doid_id"].unique())
    uniq_do_ids = [c for c in uniq_do_ids if c != "Control"]
    
    # create pseudo-label for disease + control
    def _get_pseudo_labels(adata_test,dsaid_2_dis):
        # get dsaid & disease labels
        dsaids_qry = adata_test.obs["dsaid"]
        dis_qry = adata_test.obs["doid_id"]

        # create pseudo-label
        # gives us specific key for controls for each disease
        pseudo_label = [f"{dsaid_2_dis[dsaid]}-Case" if dis != "Control" else f"{dsaid_2_dis[dsaid]}-{dis}" for dsaid, dis in zip(dsaids_qry, dis_qry)]

        return pseudo_label


    # create pseudo-labels to know which controls belong to which diseases
    df_qry_obs = adata_qry.obs.copy()
    df_qry_obs["pseudo_label"] = _get_pseudo_labels(adata_qry,dsaid_2_dis)

    df_ref_obs = adata_ref.obs.copy()

    # same disease / control within adata
    s_same_diff = list()
    l_same_diff = list()

    s_ctrl_diff = list()
    l_ctrl_diff = list()

    # same disease across adatas
    for doid_i in tqdm(uniq_do_ids):

        datasets_ref = df_ref_obs.query(f"doid_id == '{doid_i}'")["dataset"].unique()
        datasets_qry = df_qry_obs.query(f"doid_id == '{doid_i}'")["dataset"].unique()
        datasets = set(datasets_ref).union(set(datasets_qry))

        if len(datasets)> 1:
            # loop through all combinations of datasets
            for ds1, ds2 in itertools.product(datasets_qry, datasets_ref):  # product of both datasets
                if ds1 == ds2:  # if from same dataset skip!
                    continue
                
                # retrieve disease idxs pairs
                idx1 = np.array(df_qry_obs.query(f"(dataset == '{ds1}') & (pseudo_label == '{doid_i}-Case')").index).astype(int)    # query disease 
                idx2 = np.array(df_qry_obs.query(f"(dataset == '{ds1}') & (pseudo_label == '{doid_i}-Control')").index).astype(int)     # query control
                idx3 = np.array(df_ref_obs.query(f"(dataset == '{ds2}') & (doid_id == '{doid_i}')").index).astype(int)              # reference disease

                # qry dis -> ref dis
                idx_pairs = np.array(list(itertools.product(idx1, idx3)))
                if len(idx_pairs) > 0:
                    s_same_diff_i = s_matrix[idx_pairs[:, 0], idx_pairs[:, 1]]
                    s_same_diff.extend(s_same_diff_i)
                    l_same_diff.extend([doid_i] * len(idx_pairs))
                    # print(len(s_same_diff_i),np.mean(s_same_diff_i), np.median(s_same_diff_i))

                # qry ctrl -> ref dis
                idx_pairs = np.array(list(itertools.product(idx2, idx3)))
                if len(idx_pairs) > 0:
                    s_ctrl_diff_i = s_matrix[idx_pairs[:, 0], idx_pairs[:, 1]]
                    s_ctrl_diff.extend(s_ctrl_diff_i)
                    l_ctrl_diff.extend([doid_i] * len(idx_pairs))

    # convert labels to weights
    w_same_diff = _convert_labels_to_weights(l_same_diff)
    w_ctrl_diff = _convert_labels_to_weights(l_ctrl_diff)

    return (s_same_diff, s_ctrl_diff), (w_same_diff, w_ctrl_diff)

def get_unrelated_pairs_split(
    adata_qry: pd.DataFrame,
    adata_ref: pd.DataFrame,
    df_related: pd.DataFrame,
    s_matrix: np.ndarray,
    n_samples: int = 1000,
    seed: int = 42,
) -> List[Tuple[int, int]]:
    """Get unrelated pairs of samples across splits"""

    np.random.seed(seed)

    adata_ref = adata_ref.copy()
    adata_ref.reset_index(drop=True, inplace=True)
    adata_ref["is_control"] = ["C" if i.startswith("Control") else "D" for i in adata_ref["doid_id"]]

    adata_qry = adata_qry.copy()
    adata_qry.reset_index(drop=True, inplace=True)
    adata_qry["is_control"] = ["C" if i.startswith("Control") else "D" for i in adata_qry["doid_id"]]
    
    # Precompute DOID-to-sample indices
    doid_to_indices = dict()
    for idx, doid in adata_ref["doid_id"].items():
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
        # idxs1 = adata_ref.query("doid_id == @d1").index
        # idxs2 = adata_ref.query("doid_id == @d2").index
        idxs1 = adata_qry.query("doid_id == @d2").index
        idxs2 = adata_ref.query("doid_id == @d1").index

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

def get_unrelated_pairs_split_2(adata_qry, adata_ref, c_matrix, df_unrelated, dsaid_2_dis:dict,sample_size:int=None, include_control:bool=False):

    fmt = EngFormatter(places=1)

    # select datasets & do_ids
    do_ids_qry = adata_qry.obs["do_id"].values
    datasets_qry = adata_qry.obs["dataset"].values

    do_ids_ref = adata_ref.obs["do_id"].values
    datasets_ref = adata_ref.obs["dataset"].values
    
    # get pairs
    unrelated_pairs = list(df_unrelated["pair_sorted"].unique())
    if include_control:
        # get pseudo label values
        do_ids_qry =  [dsaid_2_dis.get(d) for d in adata_qry.obs["dsaid"]]
        do_ids_ref =  [dsaid_2_dis.get(d) for d in adata_ref.obs["dsaid"]]


    unrelated_universe = get_doid_idxs_pairs_split(unrelated_pairs, do_ids_qry, do_ids_ref, datasets_qry, datasets_ref, return_weights=False)
    
    if sample_size:    
        # get sample of unrelated pairs
        _idxs = random.sample(range(len(unrelated_universe)), sample_size)
        unrelated_universe = unrelated_universe[_idxs]

    s_unrelated = c_matrix[unrelated_universe[:, 0], unrelated_universe[:, 1]]
    w_unrelated = np.ones(len(s_unrelated))   # do not weight - they are random - no groupings
    l_unrelated = f"Unrelated | {fmt(len(s_unrelated))} comps"

    return s_unrelated, w_unrelated, l_unrelated

def get_doid_idxs_pairs(pairs:List[Tuple], do_ids:List, datasets:List=None, return_weights:bool = False)-> Tuple[List[Tuple[int, int]], List[int]]:

    universe_pairs = list()
    l_weights = list()  # label weights
    for doid_a, doid_b in tqdm(pairs):
        
        # retrieve disease idxs pairs
        idx1 = np.where(do_ids == doid_a)[0]
        idx2 = np.where(do_ids == doid_b)[0]

        # compute idx combinations
        idx_pairs = list(itertools.product(idx1, idx2))

        # if ONLY take into account diseases DIFFERENT datasets
        #! NOTE - BEST WAY TO REMOVE SAME DATASETS - SHOULD BE APPLIED ALWAYS
        if datasets is not None:
            # filter by datasets
            idx_pairs = [p for p in idx_pairs if datasets[p[0]] != datasets[p[1]]]

        universe_pairs.extend(idx_pairs)

        if return_weights:
            l_weights.extend([f"{doid_a} - {doid_b}"] * len(idx_pairs)) # joint label

    universe_pairs = np.array(universe_pairs).reshape(-1, 2)
    if return_weights:
        return universe_pairs, l_weights
    else:
        return universe_pairs
    
def get_doid_idxs_pairs_split(pairs:List[Tuple], do_ids_1:List, do_ids_2:List, datasets_1:List=None, datasets_2:List=None, do_diff_dt:bool=True, return_weights:bool = False)-> Tuple[List[Tuple[int, int]], List[int]]:

    # make sure do_ids_1 & do_ids_2 are arrays
    do_ids_1 = np.array(do_ids_1)
    do_ids_2 = np.array(do_ids_2)

    universe_pairs = list()
    l_weights = list()  # label weights

    for doid_a, doid_b in tqdm(pairs):
        # retrieve disease idxs pairs
        # b - a
        idx1_1 = np.where(do_ids_1 == doid_a)[0]    # idxs disease a in split 1
        idx1_2 = np.where(do_ids_2 == doid_b)[0]    # idxs disease b in split 2

        # a - b
        idx2_1 = np.where(do_ids_1 == doid_b)[0]    # idxs disease b in split 1
        idx2_2 = np.where(do_ids_2 == doid_a)[0]    # idxs disease a in split 2

        # compute idx combinations disease a 
        idx_pairs_a_b = list(itertools.product(idx1_1, idx1_2))
        idx_pairs_b_a = list(itertools.product(idx2_1, idx2_2))

        # if ONLY take into account diseases DIFFERENT datasets
        #! NOTE - BEST WAY TO REMOVE SAME DATASETS - SHOULD BE APPLIED ALWAYS
        if do_diff_dt:
            # filter by datasets
            idx_pairs_a_b = [p for p in idx_pairs_a_b if datasets_1[p[0]] != datasets_2[p[1]]]
            idx_pairs_b_a = [p for p in idx_pairs_b_a if datasets_1[p[0]] != datasets_2[p[1]]]

        universe_pairs.extend(idx_pairs_a_b)
        universe_pairs.extend(idx_pairs_b_a)

        if return_weights:
            l_weights.extend([f"{doid_a} - {doid_b}"] * len(idx_pairs_a_b)) # joint label
            l_weights.extend([f"{doid_a} - {doid_b}"] * len(idx_pairs_b_a)) # joint label

    universe_pairs = np.array(universe_pairs).reshape(-1, 2)
    if return_weights:
        return universe_pairs, l_weights
    else:
        return universe_pairs


def get_doid_idxs_single(do_ids:List, datasets:List=None, return_weights:bool = False)-> Tuple[List[Tuple[int, int]], List[int]]:

    # remove controls
    uniq_do_ids = list(set(do_ids))
    uniq_do_ids = [c for c in uniq_do_ids if c != "Control"]

    print(f"Nº of diseases: {len(uniq_do_ids)}")

    universe_pairs = list()
    l_weights = list()  # label weights
    for doid_i in tqdm(uniq_do_ids):

        # retrieve disease idxs pairs
        idx1 = np.where(do_ids == doid_i)[0]

        # compute idx combinations
        idx_pairs = list(itertools.combinations(idx1, 2))

        # if ONLY take into account diseases DIFFERENT datasets
        if datasets is not None:
            # filter by datasets
            idx_pairs = [p for p in idx_pairs if datasets[p[0]] != datasets[p[1]]]

        universe_pairs.extend(idx_pairs)

        if return_weights:
            l_weights.extend([doid_i] * len(idx_pairs))

    universe_pairs = np.array(universe_pairs).reshape(-1, 2)
    if return_weights:
        return universe_pairs, l_weights
    else:
        return universe_pairs


def get_doid_idxs_single_set(do_ids_1:List,do_ids_2:List, datasets_1:List, datasets_2:List=None, do_diff_dt:bool = True, return_weights:bool = False)-> Tuple[List[Tuple[int, int]], List[int]]:

    uniq_do_ids = list(set(do_ids_1))
    uniq_do_ids = [c for c in uniq_do_ids if c != "Control"]

    print(f"Nº of diseases: {len(uniq_do_ids)}")

    universe_pairs = list()
    l_weights = list()  # label weights
    for doid_i in tqdm(uniq_do_ids):

        # retrieve disease idxs pairs
        idx1 = np.where(do_ids_1 == doid_i)[0]
        idx2 = np.where(do_ids_2 == doid_i)[0]

        # compute idx combinations
        idx_pairs = list(itertools.product(idx1, idx2))

        # if ONLY take into account diseases DIFFERENT datasets
        if do_diff_dt:
            # filter by datasets
            idx_pairs = [p for p in idx_pairs if datasets_1[p[0]] != datasets_2[p[1]]]

        universe_pairs.extend(idx_pairs)

        if return_weights:
            l_weights.extend([doid_i] * len(idx_pairs))

    universe_pairs = np.array(universe_pairs).reshape(-1, 2)
    if return_weights:
        return universe_pairs, l_weights
    else:
        return universe_pairs

def get_related_dis_sim(
    adata, c_matrix, df_related, df_unrelated, labels, sample:int=None
):
    fmt = EngFormatter(places=1)
   
   # select datasets & do_ids
    do_ids = adata.obs["do_id"].values
    datasets = adata.obs["dataset"].values

    # get pairs    
    unrelated_pairs = list(df_unrelated["pair_sorted"].unique())
    unrelated_universe = get_doid_idxs_pairs(unrelated_pairs, do_ids, return_weights=False)
    print(f"Nº of unrelated disease pairs: {len(unrelated_universe)}")
    
    c_all = list()
    w_all = list()
    l_all = list()
    _sample_sizes = list()
    for i, label_i in enumerate(labels):
        _df_related = df_related.query(f"shortest_path_length == {label_i}")
        print(f"[SP {label_i}] Nº of related disease pairs: {len(_df_related)}")

        # get related pairs
        related_idxs, _l_related = get_doid_idxs_pairs(list(_df_related["pair_sorted"]), do_ids, datasets, return_weights=True)
        print(f"Nº of related disease pairs for {label_i}: {len(related_idxs)}")

        # convert label to weights - all pairs add to 1
        _w_related = _convert_labels_to_weights(_l_related)

        if sample:
            _sample_i = random.sample(range(len(related_idxs)), sample)
            related_idxs = related_idxs[_sample_i]
            _w_related = _w_related[_sample_i]

        # retrieve similarities
        _c_related = c_matrix[related_idxs[:, 0], related_idxs[:, 1]]

        # append values
        c_all.append(_c_related)
        w_all.append(_w_related)
        _sample_sizes.append(len(_c_related))

        # append labels
        l_all.append(f"{label_i} | {len(_df_related)} pairs | {fmt(len(_c_related))} comps",)


    # get sample of SAME DISEASE
    same_idxs, _l_same = get_doid_idxs_single(do_ids, datasets, return_weights=True)
    _w_same = _convert_labels_to_weights(_l_same)
    _c_same = c_matrix[same_idxs[:, 0], same_idxs[:, 1]]
    c_all.append(_c_same)
    w_all.append(_w_same)
    l_all.append(f"Same Disease | {len(set(do_ids)-set(['Control']))} dis | {fmt(len(_c_same))} comps")

    # get sample of unrelated pairs
    _sample_i = random.sample(range(len(unrelated_universe)), max(_sample_sizes))
    _unrelated_sample = unrelated_universe[_sample_i]
    print(f"Nº of unrelated disease pairs: {len(_unrelated_sample)}")
    
    _s_unrelated = c_matrix[_unrelated_sample[:, 0], _unrelated_sample[:, 1]]
    _w_unrelated = np.ones(len(_s_unrelated))   # do not weight - they are random - no groupings
    c_all.append(_s_unrelated)
    w_all.append(_w_unrelated)
    l_all.append(f"Unrelated | {fmt(len(_s_unrelated))} comps")

    return c_all, w_all, l_all





def get_related_dis_sim_splits(
    adata_qry, adata_ref, c_matrix, df_related, df_unrelated, labels, sample:int=None
):
    fmt = EngFormatter(places=1)
   
   # select datasets & do_ids
    do_ids_qry = adata_qry.obs["do_id"].values
    datasets_qry = adata_qry.obs["dataset"].values

    do_ids_ref = adata_ref.obs["do_id"].values
    datasets_ref = adata_ref.obs["dataset"].values

    # get pairs    
    unrelated_pairs = list(df_unrelated["pair_sorted"].unique())
    unrelated_universe = get_doid_idxs_pairs_split(unrelated_pairs, do_ids_qry, do_ids_ref, datasets_qry, datasets_ref, return_weights=False)
    print(f"Nº of unrelated disease pairs: {len(unrelated_universe)}")
    
    c_all = list()
    w_all = list()
    l_all = list()
    _sample_sizes = list()
    for _, label_i in enumerate(labels):
        _df_related = df_related.query(f"shortest_path_length == {label_i}")
        print(f"[SP {label_i}] Nº of related disease pairs: {len(_df_related)}")

        # get related pairs
        related_idxs, _l_related = get_doid_idxs_pairs_split(list(_df_related["pair_sorted"]), do_ids_qry, do_ids_ref, datasets_qry, datasets_ref, return_weights=True)
        print(f"Nº of related disease pairs for {label_i}: {len(related_idxs)}")

        # convert label to weights - all pairs add to 1
        _w_related = _convert_labels_to_weights(_l_related)

        if sample:
            _sample_i = random.sample(range(len(related_idxs)), sample)
            related_idxs = related_idxs[_sample_i]
            _w_related = _w_related[_sample_i]

        # retrieve similarities
        _c_related = c_matrix[related_idxs[:, 0], related_idxs[:, 1]]

        # append values
        c_all.append(_c_related)
        w_all.append(_w_related)
        _sample_sizes.append(len(_c_related))

        # append labels
        l_all.append(f"{label_i} | {len(_df_related)} pairs | {fmt(len(_c_related))} comps",)


    # get sample of SAME DISEASE
    same_idxs, _l_same = get_doid_idxs_single_set(do_ids_qry, do_ids_ref, datasets_qry, datasets_ref, return_weights=True)    
    _w_same = _convert_labels_to_weights(_l_same)
    if sample:
        _sample_i = random.sample(range(len(same_idxs)), sample)
        same_idxs = same_idxs[_sample_i]
        _w_same = _w_same[_sample_i]

    _c_same = c_matrix[same_idxs[:, 0], same_idxs[:, 1]]
    c_all.append(_c_same)
    w_all.append(_w_same)
    l_all.append(f"Same Disease | {len(set(do_ids_qry)-set(['Control']))} dis | {fmt(len(_c_same))} comps")

    # get sample of unrelated pairs
    _sample_i = random.sample(range(len(unrelated_universe)), max(_sample_sizes))
    _unrelated_sample = unrelated_universe[_sample_i]
    print(f"Nº of unrelated disease pairs: {len(_unrelated_sample)}")
    
    _s_unrelated = c_matrix[_unrelated_sample[:, 0], _unrelated_sample[:, 1]]
    _w_unrelated = np.ones(len(_s_unrelated))   # do not weight - they are random - no groupings
    c_all.append(_s_unrelated)
    w_all.append(_w_unrelated)
    l_all.append(f"Unrelated | {fmt(len(_s_unrelated))} comps")

    return c_all, w_all, l_all

def transform_tsne(adata, sample_size=None):
    adata_copy = adata.copy()

    def _fill_nans(X):
        X[np.isnan(adata_copy.X)] = 0
        return X

    if sample_size and adata_copy.n_obs > sample_size:
        # pick random observation indices (without replacement)
        _idxs = np.random.choice(len(adata_copy), size=min(sample_size, len(adata_copy)), replace=False)

        # boolean mask
        mask = np.zeros(len(adata_copy), dtype=bool)
        mask[_idxs] = True

        # subset and make an explicit copy (AnnData best practice)
        adata_copy = adata_copy[mask].copy()

    # get raw gene expression
    X = np.array(adata_copy.X).copy()

    # fill nans
    X = _fill_nans(X)

    # compute t-SNE on expression matrix
    X = TSNE(n_components=2, random_state=42).fit_transform(X)
    return X, adata_copy

def plot_tsne_adata(X:np.array, adata: sc.AnnData, label: str, dpi:int=300, title:str=None) -> None:
    
    # plot 
    lab = adata.obs[label]
    mask_control = (lab == "Control") | (lab == "unknown") | (lab == "nan")
    X_non_control = X[~mask_control]
    lab_nc = lab[~mask_control]
    X_control = X[mask_control]
    lab_c = lab[mask_control]
    plt.figure(figsize=(3,3), dpi=dpi)   

    if len(X_control) > 0:
        plt.scatter(X_control[:,0], X_control[:,1],
                    c="lightgrey", s=27, alpha=0.9, label="Control", linewidths=0)


    # get colors for labels    
    if not pd.api.types.is_categorical_dtype(lab):
        lab = lab.astype("category")

    codes = lab_nc.cat.codes.to_numpy()
    cats  = lab_nc.cat.categories.to_list()
    cmap = cm.get_cmap("tab20", len(cats)) if len(cats) <= 20 else cm.get_cmap("gist_rainbow", len(cats))

    sc_plot = plt.scatter(X_non_control[:,0], X_non_control[:,1],
                        c=codes, cmap=cmap, s=20, alpha=0.7,linewidths=0)

    plt.title(f"t-SNE colored by {label}")

    if title:
        plt.title(title)

    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.tight_layout()
    
    # remove legend
    if len(cats) <= 20:
        cbar = plt.colorbar(sc_plot, ticks=range(len(cats)))
        cbar.ax.set_yticklabels(cats)
    plt.show()

