# imports
import argparse
import copy
import json
import logging
import os
import pickle
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import *

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import seaborn as sns
import torch
import wandb
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind

# ===== Local paths (add before importing scgpt) =====
sys.path.insert(0, "/aloy/home/ddalton/git_clones/scGPT")
sys.path.insert(0, "../")
sys.path.append("/aloy/home/ddalton/projects/scGPT_playground/")

# ===== scGPT =====
import scgpt as scg
from scgpt import SubsetsBatchSampler
from scgpt.loss import (
    criterion_neg_log_bernoulli,
    masked_mse_loss,
    masked_relative_error,
)
from scgpt.model import AdversarialDiscriminator, TransformerModel
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import random_mask_value, tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import category_str2int, eval_scib_metrics, set_seed
from scgpt.tasks.cell_emb import get_batch_cell_embeddings

# ===== Project helpers =====
from scanpy.pp import combat
from src.training import helpers as tr_h
from src.utils import viz as vz

# ===== One-time setup =====
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings("ignore")

print(f"scGPT version: {scg.__file__}")
logging.info(f"Is Cuda Available {torch.cuda.is_available()}")

print(torch.cuda.device_count())  # Check how many GPUs are available
parser = argparse.ArgumentParser(description="Script for scGPT project")

# variables
run_dir = "/aloy/home/ddalton/projects/scGPT_playground/outputs/run-25-09-17-01"
# run_dir = "/aloy/home/ddalton/projects/scGPT_playground/outputs/run-25-09-13-18"
# run_dir = "/aloy/home/ddalton/projects/scGPT_playground/outputs/run-25-09-28-02"
set_cls_to_pad = False
include_zero_gene = False
pad_nan_vals = False
is_processed= False

query_data_path = "/aloy/home/ddalton/projects/scGPT_playground/data/pp_data-25-09-12-01/data.h5ad"
# query_data_path = os.path.join(run_dir, "adata_test_1.h5ad")
data_name = "microarray"
model_type = "ft"
assert model_type in ["ft", "pt"], "model_type must be 'ft' or 'pt'"

manual_parameters = {
    "scgpt_pp": "norm_log1p",  # options: raw, norm_log1p, binned
}
mask_value = -1
pad_value = -2
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]


d_input_layer = {  # the values of this map coorespond to the keys in preprocessing
                "normed_raw": "X_normed",
                "log1p": "X_normed",
                "binned": "X_binned",
                }

# functions
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

def generate_scratch_folder(run_dir:str)-> str:
    # get run name
    run_name = run_dir.split("/")[-1]

    # base directory in scratch
    base_output_dir = "/aloy/scratch/ddalton/projects/scGPT_playground/outputs"

    # define & create output directory
    output_dir = os.path.join(base_output_dir,run_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "outputs"), exist_ok=True)

    print(f"Output directory created: {output_dir}")
    return output_dir


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

def get_embeddings(model: nn.Module,
                   data_loader: DataLoader,
                   vocab: Vocab,
                   device: torch.device) -> np.array:
    embeddings = []
    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="Embedding (finetuned model)"):
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])

            # fp16/bf16 autocast so flash-attn accepts the dtype
            with torch.cuda.amp.autocast(enabled=True):
                hidden = model._encode(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                )

            cls_emb = hidden[:, 0, :].cpu().numpy()
            embeddings.append(cls_emb)
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings

def get_loader(adata: AnnData,
                   batch_size: int,
                   vocab: Vocab,
                   max_length: int,
                   include_zero_gene: bool = True,
                   set_cls_to_pad: bool = True,
                    pad_nan_vals: bool = True,
                   input_layer_key: str = "X_binned",
                   is_processed: bool = True,
                   pad_token:str = "<pad>",
                   pad_value: float = -2,
                   mask_ratio: float = 0.0,
                   mask_value:float = -1,
                   n_bins:int=51) -> DataLoader:


    # generate vocab
    genes = adata_query.var["gene_name"].tolist()
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)

    if not is_processed:
        
        adata.X = adata.X.toarray() if issparse(adata.X) else adata.X

        # set up preprocessor
        preprocessor = Preprocessor(
            use_key="X",  # the key in adata.layers to use as raw data
            filter_gene_by_counts=False,  # step 1
            filter_cell_by_counts=False,  # step 2 #! WE HAVE CASES WHERE EVERYTHING IS 0 - WE SHOULD ACTIVATE THIS!
            normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
            result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
            log1p=True,  # 4. whether to log1p the normalized data
            result_log1p_key="X_log1p",
            subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
            hvg_flavor="seurat_v3" if True else "cell_ranger",
            binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
            result_binned_key="X_binned",  # the key in adata.layers to store the binned data
            )

        # preprocess
        mask_nans = np.isnan(adata.X)
        adata.X[mask_nans] = 0.0    # set to 0
   
        preprocessor(adata, batch_key=None)
   
        if pad_nan_vals:
            adata.layers[input_layer_key][mask_nans] = pad_value

        print(f"Min value in input: {adata.layers[input_layer_key].min()}")

    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )

    # celltypes_labels = np.array(adata.obs["celltype_id"].tolist())
    # batch_ids = np.array(adata.obs["batch_id"].tolist())

    tokenized_data = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=max_length,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=include_zero_gene,
    )

    # define values for cls token
    if set_cls_to_pad:
        tokenized_data["values"][:, 0] = pad_value

    input_values = random_mask_value(
        tokenized_data["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )

    data = {
        "gene_ids": tokenized_data["genes"],
        "values": input_values,
        "target_values": tokenized_data["values"],
        # "batch_labels": torch.from_numpy(batch_ids).long(),
        # "celltype_labels": torch.from_numpy(celltypes_labels).long(),
    }

    data_loader = DataLoader(
        dataset=SeqDataset(data),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)), batch_size // 2),
        pin_memory=True,
    )
    return data_loader

# load variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_style = "binned"  # "normed_raw", "log1p", or "binned"

batch_size = 16
eval_batch_size = 16
max_seq_len = 3501

# load adata train
adata_model = sc.read(os.path.join(run_dir, "adata_train_1.h5ad"))

# load data
adata_query = sc.read(query_data_path)
# adata_query = adata_query[adata_query.obs["library"] == "RNA-Seq"].copy()
# print("Using RNA-Seq data only - adata shape", adata_query.shape)

# generate output folder
output_dir = generate_scratch_folder(run_dir)

if model_type == "ft":
    model_config_file = os.path.join(run_dir,"args.json")
    vocab_file = os.path.join(run_dir, "vocab.json")

    vocab = GeneVocab.from_file(vocab_file)

    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    adata_query.var["id_in_vocab"] = [
            vocab[gene] if gene in vocab else -1 for gene in adata_query.var["gene_name"]
        ]

    gene_ids_in_vocab = np.array(adata_query.var["id_in_vocab"])
    print(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )

    # model
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)

    #! TEST-OUT
    # def subset_adata(adata, adata_ref):
    #     import numpy as np
    #     import pandas as pd

    #     # mask presence in ref
    #     mask = np.isin(adata.obs.ids, adata_ref.obs.ids)
    #     adata_subset = adata[mask]

    #     # re-order to match
    #     desired = adata_ref.obs["ids"].tolist()
    #     cat = pd.Categorical(adata_subset.obs["ids"], categories=desired, ordered=True)
    #     adata_subset = adata_subset[adata_subset.obs.assign(_k=cat).sort_values("_k").index, :].copy()

    #     assert (adata_subset.obs["ids"].values == adata_ref.obs["ids"].values).all()

    #     return adata_subset

    # adata_query = subset_adata(adata_query, adata_model)
    # print("After subsetting to match valid data:", adata_query.shape)

    # # filter genes
    # mask_genes = np.isin(adata_query.var["gene_name"].tolist(), adata_model.var["gene_name"].tolist() )
    # adata_query = adata_query[:, mask_genes]
    # print("After subsetting to match valid genes:", adata_query.shape)

    import sys
    sys.path.append("../..")
    from src.utils import utils as ut

    # subset by disease
    df_info = ut.load_dsa_info()

    # map dsaid to disease id
    dsa_to_disease_id = dict(zip(df_info["dsaid"], df_info["diseaseid"]))

    # disease ids to keep
    _dsaid = adata_model.obs["dsaid"].to_list()
    disease_ids = [dsa_to_disease_id[d] for d in _dsaid]
    print(f"Nº of diseases : {len(set(disease_ids))}")

    # get all dsaids w/ said diease ids
    all_dsaids = [d for d, v in dsa_to_disease_id.items() if v in disease_ids]
    print(f"Nº of dsaids : {len(set(_dsaid))}")

    # filter adata
    adata_query = adata_query[adata_query.obs["library"] == "RNA-Seq"].copy()
    adata_query = adata_query[adata_query.obs["dsaid"].isin(all_dsaids)].copy()
    # print("Using Microarray data only - adata shape", adata_query.shape)
    print("Using RNA-Seq data only - adata shape", adata_query.shape)
    print(f"Nº of diseases : {len(set(adata_query.obs['doid_id'].to_list()))}")

    # mask samples
    nan_thr= 0.5

    non_nan_mask = ~np.isnan(adata_query.X)  & ~(adata_query.X==0) 
    non_nan_mask_pct = np.sum(non_nan_mask, axis=1) / adata_query.X.shape[1]

    # mask samples that have less than 30% non-NaN values
    mask_samples_nan = non_nan_mask_pct >= nan_thr 

    print(f"{nan_thr} Keeping {np.sum(mask_samples_nan)} samples out of {adata_query.X.shape[0]} ({np.sum(mask_samples_nan)/adata_query.X.shape[0]*100:.2f}%)")

    zero_thr = 0.5
    non_zero_mask = ~(adata_query.X==0) 
    non_zero_mask_pct = np.sum(non_zero_mask, axis=1) / adata_query.X.shape[1]

    # mask samples that have less than 30% non-NaN values
    mask_samples_zero = non_zero_mask_pct >= zero_thr 

    print(f"{zero_thr} Keeping {np.sum(mask_samples_zero)} samples out of {adata_query.X.shape[0]} ({np.sum(mask_samples_zero)/adata_query.X.shape[0]*100:.2f}%)")

    mask_samples_comb = mask_samples_nan & mask_samples_zero
    print(f"Combined: Keeping {np.sum(mask_samples_comb)} samples out of {adata_query.X.shape[0]} ({np.sum(mask_samples_comb)/adata_query.X.shape[0]*100:.2f}%)")

    # apply the mask to the AnnData object
    adata_query = adata_query[mask_samples_comb, :]

    input_layer_key = d_input_layer[input_style]

    # Load the pre-trained scGPT model
    test_loader = get_loader(
        adata=adata_query,
        vocab=vocab,
        batch_size=batch_size,
        max_length=max_seq_len,
        include_zero_gene=include_zero_gene,
        set_cls_to_pad=set_cls_to_pad,
        pad_nan_vals=pad_nan_vals,
        input_layer_key=input_layer_key,
        is_processed=is_processed,
        pad_token=pad_token,
        pad_value=pad_value,
        mask_ratio=0.0,  # no masking during evaluation
        mask_value=mask_value,
    ) 

    # load finetuned model
    model_path = os.path.join(run_dir, "model_1.pt")
    model_ft = torch.load(model_path, map_location=device)  # try to read container
    model_ft.to(device)
    model_ft.eval()


    # generate embeddings
    embeddings = get_embeddings(model_ft, test_loader, vocab, device)

    # save embedding
    pickle.dump(embeddings, open(os.path.join(output_dir, f"{data_name}-{model_type}.pkl"), "wb"))


elif model_type == "pt":

    model_dir = Path("/aloy/home/ddalton/projects/scGPT_playground/save/scGPT_human")
    model_config_file = model_dir / "args.json"

    with open(model_config_file, "r") as f:
        model_configs = json.load(f)

    sc.pp.normalize_total(adata_query, target_sum=1e4) 
    sc.pp.log1p(adata_query)

    cell_embeddings = scg.tasks.embed_data(
            adata_query,
            model_dir,
            gene_col="gene_name",
            batch_size=64,
            max_length=max_seq_len,
            do_binning=True,
        )
    pickle.dump(cell_embeddings, open(os.path.join(output_dir, f"{data_name}-{model_type}.pkl"), "wb"))
