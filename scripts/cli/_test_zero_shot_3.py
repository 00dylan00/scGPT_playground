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
method = 3
run_dir = "/aloy/home/ddalton/projects/scGPT_playground/outputs/run-25-09-23-03"
# query_data_path = "/aloy/home/ddalton/projects/scGPT_playground/data/pp_data-25-09-12-01/data.h5ad"
query_data_path = os.path.join(run_dir, "adata_test_1.h5ad")
data_name = "test"
is_processed= True

manual_parameters = {
    "scgpt_pp": "norm_log1p",  # options: raw, norm_log1p, binned
}
mask_value = -1
pad_value = -2
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
gene_col = "gene_name"

max_seq_len = 3501
include_zero_gene = False
mask_ratio = 0.0
eval_batch_size = 16
batch_size = eval_batch_size
device = "cuda"
model_dir = "/aloy/home/ddalton/projects/scGPT_playground/save/scGPT_human"
input_style = "binned"

use_fast_transformer = True

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

output_dir = generate_scratch_folder(run_dir)

# load data
adata_query = sc.read(query_data_path)
adata_query = adata_query[adata_query.obs["library"] == "RNA-Seq"].copy()
print("Using RNA-Seq data only - adata shape", adata_query.shape)

# Method 1
if method == 1:
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

    if config.get("load_model") is not None:
        model_dir = Path(config.get("load_model"))
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"

        vocab = GeneVocab.from_file(vocab_file)

        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)

        # adata_query.var["id_in_vocab"] = [
        #     1 if gene in vocab else -1 for gene in adata_query.var["gene_name"]
        # ]
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

    # load finetuned model
    model_path = os.path.join(run_dir, "best_model.pt")
    model_ft = torch.load(model_path, map_location=device)  # try to read container
    model_ft.to(device)
    model_ft.eval()


    embeddings_pt_1 = []
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Embedding (finetuned model)"):
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])

            # fp16/bf16 autocast so flash-attn accepts the dtype
            with torch.cuda.amp.autocast(enabled=config.get("amp", True)):
                hidden = model._encode(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                )

            cls_emb = hidden[:, 0, :].cpu().numpy()
            embeddings_pt_1.append(cls_emb)

    embeddings_pt_1 = np.concatenate(embeddings_pt_1, axis=0)

    # save
    pickle.dump(embeddings_pt_1, open(os.path.join(output_dir, f"method_{method}-{data_name}-pt_all_outputs_2.pkl"), "wb"))


# Method 2
if method == 2:
    adata_or_file = adata_query
    if isinstance(adata_or_file, AnnData):
        adata = adata_or_file
    else:
        adata = sc.read_h5ad(adata_or_file)

    # verify gene col
    if gene_col == "index":
        adata.var["index"] = adata.var.index
    else:
        assert gene_col in adata.var

    if device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available. Using CPU instead.")

    # LOAD MODEL
    model_dir = Path(model_dir)
    vocab_file = model_dir / "vocab.json"
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]

    # vocabulary
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    adata.var["id_in_vocab"] = [
        vocab[gene] if gene in vocab else -1 for gene in adata.var[gene_col]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    print(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    with open(model_config_file, "r") as f:
        model_configs = json.load(f)

    # Binning will be applied after tokenization. A possible way to do is to use the unified way of binning in the data collator.

    vocab.set_default_index(vocab["<pad>"])
    genes = adata.var[gene_col].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)

    # all_counts = adata.layers["counts"]
    # num_of_non_zero_genes = [
    #     np.count_nonzero(all_counts[i]) for i in range(all_counts.shape[0])
    # ]
    # max_length = min(max_length, np.max(num_of_non_zero_genes) + 1)



    # model = TransformerModel(
    #     ntoken=len(vocab),
    #     d_model=model_configs["embsize"],
    #     nhead=model_configs["nheads"],
    #     d_hid=model_configs["d_hid"],
    #     nlayers=model_configs["nlayers"],
    #     nlayers_cls=model_configs["n_layers_cls"],
    #     n_cls=1,
    #     vocab=vocab,
    #     dropout=model_configs["dropout"],
    #     pad_token=model_configs["pad_token"],
    #     pad_value=model_configs["pad_value"],
    #     do_mvc=True,
    #     do_dab=False,
    #     use_batch_labels=False,
    #     domain_spec_batchnorm=False,
    #     explicit_zero_prob=False,
    #     use_fast_transformer=use_fast_transformer,
    #     fast_transformer_backend="flash",
    #     pre_norm=False,
    # )

    # from scgpt.utils import load_pretrained
    # load_pretrained(model, torch.load(model_file, map_location=device), verbose=False)
    # model.to(device)
    # model.eval()


    # load finetuned model
    model_path = os.path.join(run_dir, "best_model.pt")
    model_ft = torch.load(model_path, map_location=device)  # try to read container
    model_ft.to(device)
    model_ft.eval()


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

    # # !FIX: force expression pad values to match collator ===
    # tokenized_test["values"][:, 0] = pad_value

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


    # === FIX: force expression pad values to match collator ===
    pad_id = vocab[pad_token]
    pad_mask = tokenized_test["genes"] == pad_id
    tokenized_test["values"][pad_mask] = pad_value   # overwrite 0.0 → -2.0

    # load finetuned model
    # model_path = os.path.join(run_dir, "best_model.pt")
    # model_ft = torch.load(model_path, map_location=device)  # try to read container
    # model_ft.to(device)
    # model_ft.eval()

    # cell_embeddings = np.zeros(
    #     (len(adata_query), model_configs["embsize"]), dtype=np.float32
    # )

    # with torch.no_grad():
    #     count = 0
    #     for batch_data in tqdm(test_loader, desc="Embedding (finetuned model)"):
    #         input_gene_ids = batch_data["gene_ids"].to(device)
    #         input_values = batch_data["values"].to(device)
    #         src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])

    #         # fp16/bf16 autocast so flash-attn accepts the dtype
    #         with torch.cuda.amp.autocast(enabled=config.get("amp", True)):
    #             embeddings = model._encode(
    #                 input_gene_ids,
    #                 input_values,
    #                 src_key_padding_mask=src_key_padding_mask,
    #                 batch_labels=None,
    #             )

    #             embeddings = embeddings[:, 0, :]  # get the <cls> position embedding
    #             embeddings = embeddings.cpu().numpy()
    #             cell_embeddings[count : count + len(embeddings)] = embeddings
    #             count += len(embeddings)
    #     cell_embeddings = cell_embeddings / np.linalg.norm(
    #         cell_embeddings, axis=1, keepdims=True
    #     )

    # from scgpt.tasks.cell_emb import get_batch_cell_embeddings
    # cell_embeddings = get_batch_cell_embeddings(
    #     adata,
    #     cell_embedding_mode="cls",
    #     model=model,
    #     vocab=vocab,
    #     max_length=1200,
    #     batch_size=16,
    #     model_configs=model_configs,
    #     gene_ids=gene_ids,
    #     use_batch_labels=False,
    # )

    # pickle.dump(cell_embeddings, open(os.path.join(output_dir, f"method_{method}-{data_name}-pt_all_outputs_2.pkl"), "wb"))
    


    count_matrix = adata.X
    count_matrix = (
        count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.toarray()
    )

    # gene vocabulary ids
    if gene_ids is None:
        gene_ids = np.array(adata.var["id_in_vocab"])
        assert np.all(gene_ids >= 0)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, count_matrix, gene_ids, batch_ids=None):
            self.count_matrix = count_matrix
            self.gene_ids = gene_ids
            self.batch_ids = batch_ids

        def __len__(self):
            return len(self.count_matrix)

        def __getitem__(self, idx):
            row = self.count_matrix[idx]
            nonzero_idx = np.nonzero(row)[0]
            values = row[nonzero_idx]
            genes = self.gene_ids[nonzero_idx]
            # append <cls> token at the beginning
            genes = np.insert(genes, 0, vocab["<cls>"])
            values = np.insert(values, 0, model_configs["pad_value"])
            genes = torch.from_numpy(genes).long()
            values = torch.from_numpy(values).float()
            output = {
                "id": idx,
                "genes": genes,
                "expressions": values,
            }
            if self.batch_ids is not None:
                output["batch_labels"] = self.batch_ids[idx]
            return output

    dataset = Dataset(
        count_matrix, gene_ids, None
    )

    # from scgpt.data_collator import DataCollator
    # from torch.utils.data import DataLoader, SequentialSampler
    # collator = DataCollator(
    #     do_padding=True,
    #     pad_token_id=vocab[model_configs["pad_token"]],
    #     pad_value=model_configs["pad_value"],
    #     do_mlm=False,
    #     do_binning=True,
    #     max_length=max_seq_len,
    #     sampling=True,
    #     keep_first_n_tokens=1,
    # )
    # data_loader = DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     sampler=SequentialSampler(dataset),
    #     collate_fn=collator,
    #     drop_last=False,
    #     num_workers=min(len(os.sched_getaffinity(0)), batch_size),
    #     pin_memory=True,
    # )

    data_loader = DataLoader(
        dataset=SeqDataset(test_data_pt),
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)), eval_batch_size // 2),
        pin_memory=True,
    )

    device = next(model_ft.parameters()).device
    # device = next(model.parameters()).device
    cell_embeddings = np.zeros(
        (len(dataset), model_configs["embsize"]), dtype=np.float32
    )
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        count = 0
        for data_dict in tqdm(data_loader, desc="Embedding cells"):
            input_gene_ids = data_dict["gene_ids"].to(device)
            input_values = data_dict["values"].to(device)
            # input_gene_ids = data_dict["gene"].to(device)
            # input_values = data_dict["expr"].to(device)
            # input_values = data_dict["values"].clone()  # make a copy so we can edit
            # input_values[:, 0] = 0.0                    # force <cls> expression = 0.0
            # input_values = input_values.to(device)        

            if count == 0:
                print(input_gene_ids.shape, input_values.shape)
                print("INPUT GENE IDS", input_gene_ids)
                print("INPUT VALUES", input_values)
                pickle.dump(input_values.cpu().numpy(), open(os.path.join(output_dir, f"input_values_3.pkl"), "wb"))
                pickle.dump(input_gene_ids.cpu().numpy(), open(os.path.join(output_dir, f"input_genes_3.pkl"), "wb"))
                print(f'DUMPED IN {os.path.join(output_dir, f"input_genes_3.pkl")}')
         
            # === QUICK HACK: deterministic truncation if too long ===
            # max_length = 1200
            # if input_gene_ids.shape[1] > max_length:
            #     idx = torch.arange(max_length)  # just keep first max_length tokens
            #     input_gene_ids = input_gene_ids[:, idx]
            #     input_values = input_values[:, idx]
            # === END HACK ===

            src_key_padding_mask = input_gene_ids.eq(
                vocab[model_configs["pad_token"]]
            )
            embeddings = model_ft._encode(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
            )

            embeddings = embeddings[:, 0, :]  # get the <cls> position embedding
            embeddings = embeddings.cpu().numpy()
            cell_embeddings[count : count + len(embeddings)] = embeddings
            count += len(embeddings)
    cell_embeddings = cell_embeddings / np.linalg.norm(
        cell_embeddings, axis=1, keepdims=True
    )
    pickle.dump(cell_embeddings, open(os.path.join(output_dir, f"method_{method}-{max_seq_len}-{data_name}-ft_all_outputs_4.pkl"), "wb"))

# Method 3
# alternative way to embed
if method == 3:

    # scGPT Fine-tuned Embedding
    model_path = "/aloy/home/ddalton/projects/scGPT_playground/outputs/run-25-09-23-03"
    embed_adata = scg.tasks.embed_data(
        adata_query,
        model_path,
        gene_col="gene_name",
        batch_size=64,
        max_length=max_seq_len,
    )
    pickle.dump(embed_adata, open(os.path.join(output_dir, f"method_{method}-{max_seq_len}-{data_name}-ft_embed_adata.pkl"), "wb"))


    # embed_adata = scg.tasks.embed_data(
    #     adata_query,
    #     model_dir,
    #     gene_col="gene_name",
    #     batch_size=64,
    #     max_length=max_seq_len,
    # )
    # pickle.dump(embed_adata, open(os.path.join(output_dir, f"method_{method}-{max_seq_len}-{data_name}-pt_embed_adata.pkl"), "wb"))


