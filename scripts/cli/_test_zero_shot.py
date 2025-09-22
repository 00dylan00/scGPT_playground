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
run_dir = "/aloy/home/ddalton/projects/scGPT_playground/outputs/run-25-09-19-02"
query_data_path = "/aloy/home/ddalton/projects/scGPT_playground/data/pp_data-25-09-12-01/data.h5ad"
# query_data_path = os.path.join(run_dir, "adata_valid_1.h5ad")
data_name = "zero_shot"
is_processed= False

manual_parameters = {
    "scgpt_pp": "norm_log1p",  # options: raw, norm_log1p, binned
}
mask_value = -1
pad_value = -2
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]

config = {'seed': 0, 
          'dataset_name': 'test_1', 
          'do_train': True, 
          'load_model': '/aloy/home/ddalton/projects/scGPT_playground/save/scGPT_human', 
          'mask_ratio': 0.0, 
          'epochs': 10, 
          'n_bins': 51, 
          'MVC': False, 
          'ecs_thres': 0.0, 
          'dab_weight': 0.0, 
          'lr': 0.0001, 
          'batch_size': 32, 
          'layer_size': 128, 
          'nlayers': 4, 
          'nhead': 4, 
          'dropout': 0.2, 
          'schedule_ratio': 0.9, 
          'save_eval_interval': 5, 
          'fast_transformer': True, 
          'pre_norm': False, 
          'amp': True, 
          'include_zero_gene': False, 
          'freeze': False, 
          'DSBN': False}

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

# load variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
include_zero_gene = (
    config["include_zero_gene"]
)  # if True, include zero genes among hvgs in the training
input_style = "binned"  # "normed_raw", "log1p", or "binned"
output_style = "binned"  # "normed_raw", "log1p", or "binned"manual_parameters.ADV
mask_ratio = config["mask_ratio"]
batch_size = 16
eval_batch_size = 16
max_seq_len = 3501


# load adata train
print("Loading data...")
print(os.path.join(run_dir, "adata_valid_1.h5ad"))
adata_valid = sc.read(os.path.join(run_dir, "adata_valid_1.h5ad"))
n_input_bins = n_bins = config["n_bins"]

# load data
adata_query = sc.read(query_data_path)
adata_query = adata_query[adata_query.obs["library"] == "RNA-Seq"].copy()
print("Using RNA-Seq data only - adata shape", adata_query.shape)


if not is_processed:
    # mask the genes
    genes_train = adata_valid.var['gene_name']
    mask_genes = adata_query.var['gene_name'].isin(genes_train)
    print(f"Number of genes in test set: {adata_query.shape[1]}")
    adata_query = adata_query[:, mask_genes]
    print(f"Filtered data to {adata_query.shape[0]} samples and {adata_query.shape[1]} genes")


    # mask samples
    nan_thr= 0.9

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


    # config parameters
    data_is_raw = True
    filter_gene_by_counts = False

    if manual_parameters.get("scgpt_pp") == "norm_log1p":
        #! QUICK FIX BECAUSE SCGPT IS FUCKING USELESS AND MESSES UP NANs
        # substitue NaNs with 0s
        adata_query.X = adata_query.X.toarray() if issparse(adata_query.X) else adata_query.X

        # set up the preprocessor, use the args to config the workflow
        preprocessor = Preprocessor(
            use_key="X",  # the key in adata_query.layers to use as raw data
            filter_gene_by_counts=False,  # step 1
            filter_cell_by_counts=False,  # step 2 #! WE HAVE CASES WHERE EVERYTHING IS 0 - WE SHOULD ACTIVATE THIS!
            normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
            result_normed_key="X_normed",  # the key in adata_query.layers to store the normalized data
            log1p=True,  # 4. whether to log1p the normalized data
            result_log1p_key="X_log1p",
            subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
            hvg_flavor="seurat_v3" if True else "cell_ranger",
            binning=config.get("n_bins"),  # 6. whether to bin the raw data and to what number of bins
            result_binned_key="X_binned",  # the key in adata_query.layers to store the binned data
            )

    #! WHAT IS THIS
    print("config.load_model", config.get("load_model"))
    if config.get("load_model") is not None:
        model_dir = Path(config.get("load_model"))
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"

        vocab = GeneVocab.from_file(vocab_file)

        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)

        adata_query.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in adata_query.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(adata_query.var["id_in_vocab"])
        print(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
        adata_query = adata_query[:, adata_query.var["id_in_vocab"] >= 0]

        # model
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        print(
            f"Resume model from {model_file}, the model args will override the "
            f"config {model_config_file}."
        )
        embsize = model_configs["embsize"]
        nhead = model_configs["nheads"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
        n_layers_cls = model_configs["n_layers_cls"]

    # convert batch ids to integers
    _batch_ids = adata_query.obs["batch_id"].tolist()
    num_batch_types = adata_query.obs["batch_id"].nunique()
    _remap_dict = {k: i for i, k in enumerate(sorted(set(_batch_ids)))}
    adata_query.obs["batch_id"] = np.array([_remap_dict[b] for b in _batch_ids], dtype=int)  # update the batch ids in adata_query.obs

    # seperate data

    mask_nans = np.isnan(adata_query.X)
    adata_query.X[mask_nans] = 0.0    # set to 0

    preprocessor(adata_query, batch_key=None)

    # set to pad value 
    adata_query.X[mask_nans] = pad_value

    # generate vocab
    genes = adata_valid.var["gene_name"].tolist()
    if config["load_model"] is None:
        vocab = Vocab(
            VocabPybind(genes + special_tokens, None)
        )  # bidirectional lookup [gene <-> int]
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)


    # generate celltype label
    celltype_id_labels = adata_query.obs["celltype"].astype("category").cat.codes.values
    celltypes = adata_query.obs["celltype"].unique()
    num_types = len(np.unique(celltype_id_labels))
    id2type = dict(enumerate(adata_query.obs["celltype"].astype("category").cat.categories))
    adata_query.obs["celltype_id"] = celltype_id_labels

if is_processed:
    if config.get("load_model") is not None:
        model_dir = Path(config.get("load_model"))
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"

        vocab = GeneVocab.from_file(vocab_file)

        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)

        adata_query.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in adata_query.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(adata_query.var["id_in_vocab"])
        print(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )


        # model
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        print(
            f"Resume model from {model_file}, the model args will override the "
            f"config {model_config_file}."
        )
        embsize = model_configs["embsize"]
        nhead = model_configs["nheads"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
        n_layers_cls = model_configs["n_layers_cls"]

    
    # generate vocab
    genes = adata_valid.var["gene_name"].tolist()
    if config["load_model"] is None:
        vocab = Vocab(
            VocabPybind(genes + special_tokens, None)
        )  # bidirectional lookup [gene <-> int]
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)

# region 3. Load the pre-trained scGPT model
ntokens = len(vocab)  # size of vocabulary
model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=3,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,

)

print(ntokens, embsize, nhead, d_hid, nlayers, vocab, pad_token, pad_value)

try:
    model.load_state_dict(torch.load(model_file))
    print(f"Loading all model params from {model_file}")
except:
    # only load params that are in the model and match the size
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_file)
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    for k, v in pretrained_dict.items():
        # print(f"Loading params {k} with shape {v.shape}")
        pass
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


model.to(device)

input_layer_key = d_input_layer[input_style]

all_counts = (
    adata_query.layers[input_layer_key].A
    if issparse(adata_query.layers[input_layer_key])
    else adata_query.layers[input_layer_key]
)

celltypes_labels = adata_query.obs["celltype_id"].tolist()  # make sure count from 0
celltypes_labels = np.array(celltypes_labels)

batch_ids = adata_query.obs["batch_id"].tolist()
batch_ids = np.array(batch_ids)

tokenized_test = tokenize_and_pad_batch(
    all_counts,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,  # append <cls> token at the beginning
    include_zero_gene=include_zero_gene,
)

input_values_test = random_mask_value(
    tokenized_test["values"],
    mask_ratio=mask_ratio,
    mask_value=mask_value,
    pad_value=pad_value,
)

test_data_pt = {
    "gene_ids": tokenized_test["genes"],
    "values": input_values_test,
    "target_values": tokenized_test["values"],
    "batch_labels": torch.from_numpy(batch_ids).long(),
    "celltype_labels": torch.from_numpy(celltypes_labels).long(),
}

test_loader = DataLoader(
    dataset=SeqDataset(test_data_pt),
    batch_size=eval_batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=min(len(os.sched_getaffinity(0)), eval_batch_size // 2),
    pin_memory=True,
)

model.eval()

embeddings = list()
with torch.no_grad():
    for batch_data in tqdm(test_loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        celltype_labels = batch_data["celltype_labels"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        with torch.cuda.amp.autocast(enabled=config["amp"]):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
                CLS=True, 
                CCE=False,
                MVC=False,
                ECS=False,
                # generative_training = False,
            )
            
            output_values = output_dict["cls_output"]

            preds = torch.sigmoid(output_values).cpu().numpy()
            preds_bin = (preds > 0.5).astype(int)
        
            output_dict = {
                key: value.cpu() if isinstance(value, torch.Tensor) else value
                for key, value in output_dict.items()
            }

        embeddings.append(output_dict)

# load finetuned model
model_path = os.path.join(run_dir, "model_1.pt")
model_ft = torch.load(model_path, map_location=device)  # try to read container
model_ft.to(device)
model_ft.eval()
embeddings_ft = list()
with torch.no_grad():
    for batch_data in tqdm(test_loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        celltype_labels = batch_data["celltype_labels"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        with torch.cuda.amp.autocast(enabled=config["amp"]):
            output_dict = model_ft(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
                CLS=True, 
                CCE=False,
                MVC=False,
                ECS=False,
                # generative_training = False,
            )
            
            output_values = output_dict["cls_output"]

            preds = torch.sigmoid(output_values).cpu().numpy()
            preds_bin = (preds > 0.5).astype(int)

            output_dict = {
                key: value.cpu() if isinstance(value, torch.Tensor) else value
                for key, value in output_dict.items()
            }


        embeddings_ft.append(output_dict)


# pre-process embeddings
embeddings = merge_embeddings(embeddings)
embeddings_ft = merge_embeddings(embeddings_ft)

# generate output folder
output_dir = generate_scratch_folder(run_dir)

# store adata query 
adata_query.write_h5ad(os.path.join(output_dir, f"{data_name}_adata.h5ad"))

# save embeddings
np.save(os.path.join(output_dir, f"{data_name}-pt_embeddings.npy"), embeddings)
np.save(os.path.join(output_dir, f"{data_name}-ft_embeddings.npy"), embeddings_ft)