"""Tutorial Annotation

Structure:
    1. Specify hyper-parameter setup for integration task
    2. Load and pre-process data
    3. Load the pre-trained scGPT model
    4. Finetune scGPT with task-specific objectives
    5. Inference with fine-tuned scGPT model
    6. Save output
"""

#region 0. Imports, Variables & Functions

import copy
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
# from typing import List, Tuple, Dict, Union, Optional
from typing import *
import warnings
import pandas as pd
from datetime import datetime

# from . import asyn
import pickle
import torch
from anndata import AnnData
import scanpy as sc
import scvi
import seaborn as sns
import numpy as np
import wandb
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from sklearn.metrics import confusion_matrix

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics
from sklearn.metrics import confusion_matrix


import numpy as np
from sklearn.model_selection import train_test_split
import logging


import os
from collections import Counter
import logging
import psutil

import pandas as pd
from typing import *
from sklearn.model_selection import StratifiedGroupKFold
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")



sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings("ignore")

logging.info(f"Is Cuda Available {str(torch.cuda.is_available())}")


import torch
print(torch.cuda.device_count())  # Check how many GPUs are available


# variables
# data_path = "../data/test_1/test_2.h5ad"
manual_parameters = {
"data_path": "/aloy/home/ddalton/projects/scGPT_playground/data/pp_data-24-09-24-01/data.h5ad",
"max_seq_len": 6500,
"batch_size":6,
"gene_presence_pct":0.1,
"benchmark_data": False,
"new_split" : True}

# functions
def train(model: nn.Module, loader: DataLoader) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    (
        total_loss,
        total_mse,
        total_cls,
        total_cce,
        total_mvc,
        total_ecs,
        total_dab,
        total_adv_E,
        total_adv_D,
        total_zero_log_prob,
        total_mvc_zero_log_prob,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    total_error = 0.0
    start_time = time.time()

    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        celltype_labels = batch_data["celltype_labels"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        with torch.cuda.amp.autocast(enabled=config.amp):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=(
                    batch_labels if INPUT_BATCH_LABELS or config.DSBN else None
                ),
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
                do_sample=do_sample_in_train,
                # generative_training=False
            )

            masked_positions = input_values.eq(mask_value)  # the postions to predict
            loss = 0.0
            metrics_to_log = {}
            if MLM:
                loss_mse = criterion(
                    output_dict["mlm_output"], target_values, masked_positions
                )
                loss = loss + loss_mse
                metrics_to_log = {"train/mse": loss_mse.item()}
            if explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_zero_log_prob
                metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
            if CLS:
                loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
                loss = loss + loss_cls
                metrics_to_log.update({"train/cls": loss_cls.item()})

                error_rate = 1 - (
                    (output_dict["cls_output"].argmax(1) == celltype_labels)
                    .sum()
                    .item()
                ) / celltype_labels.size(0)
            if CCE:
                loss_cce = 10 * output_dict["loss_cce"]
                loss = loss + loss_cce
                metrics_to_log.update({"train/cce": loss_cce.item()})
            if MVC:
                loss_mvc = criterion(
                    output_dict["mvc_output"], target_values, masked_positions
                )
                loss = loss + loss_mvc
                metrics_to_log.update({"train/mvc": loss_mvc.item()})
            if MVC and explicit_zero_prob:
                loss_mvc_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mvc_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_mvc_zero_log_prob
                metrics_to_log.update({"train/mvc_nzlp": loss_mvc_zero_log_prob.item()})
            if ECS:
                loss_ecs = 10 * output_dict["loss_ecs"]
                loss = loss + loss_ecs
                metrics_to_log.update({"train/ecs": loss_ecs.item()})
            if DAB:
                # try weighting and separate optimizer
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                loss = loss + dab_weight * loss_dab
                metrics_to_log.update({"train/dab": loss_dab.item()})

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        if ADV:
            # rerun the model for adversarial training
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=(
                    batch_labels if INPUT_BATCH_LABELS or config.DSBN else None
                ),
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
                do_sample=do_sample_in_train,
                # generative_training=False
            )

            # TRAINING DISCRIMINATOR
            loss_adv_D = criterion_adv(
                discriminator(output_dict["cell_emb"].detach()), batch_labels
            )
            if epoch > adv_D_delay_epochs:
                discriminator.zero_grad()
                loss_adv_D.backward()
                optimizer_D.step()

            # TRAINING ENCODER
            loss_adv_E = -criterion_adv(
                discriminator(output_dict["cell_emb"]), batch_labels
            )
            # NOTE: the loss is negative here because we want to maximize
            # the cross_entropy_loss, in other words, disguise against the discriminator
            if epoch > adv_E_delay_epochs:
                model.zero_grad()
                discriminator.zero_grad()
                loss_adv_E.backward()
                optimizer_E.step()

        wandb.log(metrics_to_log)

        total_loss += loss.item()
        total_mse += loss_mse.item() if MLM else 0.0
        total_cls += loss_cls.item() if CLS else 0.0
        total_cce += loss_cce.item() if CCE else 0.0
        total_mvc += loss_mvc.item() if MVC else 0.0
        total_ecs += loss_ecs.item() if ECS else 0.0
        total_dab += loss_dab.item() if DAB else 0.0
        total_adv_E += loss_adv_E.item() if ADV else 0.0
        total_adv_D += loss_adv_D.item() if ADV else 0.0
        total_zero_log_prob += loss_zero_log_prob.item() if explicit_zero_prob else 0.0
        total_mvc_zero_log_prob += (
            loss_mvc_zero_log_prob.item() if MVC and explicit_zero_prob else 0.0
        )
        total_error += error_rate
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_cls = total_cls / log_interval if CLS else 0.0
            cur_cce = total_cce / log_interval if CCE else 0.0
            cur_mvc = total_mvc / log_interval if MVC else 0.0
            cur_ecs = total_ecs / log_interval if ECS else 0.0
            cur_dab = total_dab / log_interval if DAB else 0.0
            cur_adv_E = total_adv_E / log_interval if ADV else 0.0
            cur_adv_D = total_adv_D / log_interval if ADV else 0.0
            cur_zero_log_prob = (
                total_zero_log_prob / log_interval if explicit_zero_prob else 0.0
            )
            cur_mvc_zero_log_prob = (
                total_mvc_zero_log_prob / log_interval
                if MVC and explicit_zero_prob
                else 0.0
            )
            cur_error = total_error / log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | "
                + (f"mse {cur_mse:5.2f} | mre {cur_error:5.2f} |" if MLM else "")
                + (f"cls {cur_cls:5.2f} | " if CLS else "")
                + (f"err {cur_error:5.2f} | " if CLS else "")
                + (f"cce {cur_cce:5.2f} |" if CCE else "")
                + (f"mvc {cur_mvc:5.2f} |" if MVC else "")
                + (f"ecs {cur_ecs:5.2f} |" if ECS else "")
                + (f"dab {cur_dab:5.2f} |" if DAB else "")
                + (f"adv_E {cur_adv_E:5.2f} |" if ADV else "")
                + (f"adv_D {cur_adv_D:5.2f} |" if ADV else "")
                + (f"nzlp {cur_zero_log_prob:5.2f} |" if explicit_zero_prob else "")
                + (
                    f"mvc_nzlp {cur_mvc_zero_log_prob:5.2f} |"
                    if MVC and explicit_zero_prob
                    else ""
                )
            )
            total_loss = 0
            total_mse = 0
            total_cls = 0
            total_cce = 0
            total_mvc = 0
            total_ecs = 0
            total_dab = 0
            total_adv_E = 0
            total_adv_D = 0
            total_zero_log_prob = 0
            total_mvc_zero_log_prob = 0
            total_error = 0
            start_time = time.time()


def define_wandb_metrcis():
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")


def evaluate(model: nn.Module, loader: DataLoader, return_raw: bool = False) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_dab = 0.0
    total_num = 0
    predictions = []
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=(
                        batch_labels if INPUT_BATCH_LABELS or config.DSBN else None
                    ),
                    CLS=CLS,  # evaluation does not need CLS or CCE
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=do_sample_in_train,
                    # generative_training = False,
                )

                output_values = output_dict["cls_output"]
                loss = criterion_cls(output_values, celltype_labels)

                if DAB:
                    loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

            total_loss += loss.item() * len(input_gene_ids)
            accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
            total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
            total_dab += loss_dab.item() * len(input_gene_ids) if DAB else 0.0
            total_num += len(input_gene_ids)
            preds = output_values.argmax(1).cpu().numpy()
            predictions.append(preds)

    wandb.log(
        {
            "valid/mse": total_loss / total_num,
            "valid/err": total_error / total_num,
            "valid/dab": total_dab / total_num,
            "valid/sum_mse_dab": (total_loss + dab_weight * total_dab) / total_num,
            "epoch": epoch,
        },
    )

    if return_raw:
        return np.concatenate(predictions, axis=0)

    return total_loss / total_num, total_error / total_num



def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    print(
        f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
        f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
    )

    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )

    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
    tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()

    tensor_celltype_labels_train = torch.from_numpy(train_celltype_labels).long()
    tensor_celltype_labels_valid = torch.from_numpy(valid_celltype_labels).long()

    if sort_seq_batch:  # TODO: update to random pick seq source in each traning batch
        train_sort_ids = np.argsort(train_batch_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]

        valid_sort_ids = np.argsort(valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
        tensor_celltype_labels_valid = tensor_celltype_labels_valid[valid_sort_ids]

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
        "celltype_labels": tensor_celltype_labels_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
        "celltype_labels": tensor_celltype_labels_valid,
    }

    return train_data_pt, valid_data_pt


# dataset
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# data_loader
def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    if num_workers == 0:
        num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)

    dataset = SeqDataset(data_pt)

    if per_seq_batch_sample:
        # find the indices of samples in each seq batch
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets,
                batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader

def test(model: nn.Module, adata: DataLoader) -> float:
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )

    celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)

    tokenized_test = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=manual_parameters.get("max_seq_len"),
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
    predictions = evaluate(
        model,
        loader=test_loader,
        return_raw=True,
    )

    # compute accuracy, precision, recall, f1
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(celltypes_labels, predictions)
    precision = precision_score(celltypes_labels, predictions, average="macro")
    recall = recall_score(celltypes_labels, predictions, average="macro")
    macro_f1 = f1_score(celltypes_labels, predictions, average="macro")

    logger.info(
        f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, "
        f"Macro F1: {macro_f1:.3f}"
    )

    results = {
        "test/accuracy": accuracy,
        "test/precision": precision,
        "test/recall": recall,
        "test/macro_f1": macro_f1,
    }

    return predictions, celltypes_labels, results

def test_2(model: nn.Module, adata: DataLoader) -> float:
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )

    celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)

    tokenized_test = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=manual_parameters.get("max_seq_len"),
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
    predictions, all_outputs = evaluate_2(
        model,
        loader=test_loader,
        device=device,
        return_raw=True,
    )

    # compute accuracy, precision, recall, f1

    accuracy = accuracy_score(celltypes_labels, predictions)
    precision = precision_score(celltypes_labels, predictions, average="macro")
    recall = recall_score(celltypes_labels, predictions, average="macro")
    macro_f1 = f1_score(celltypes_labels, predictions, average="macro")

    logger.info(
        f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, "
        f"Macro F1: {macro_f1:.3f}"
    )

    results = {
        "test/accuracy": accuracy,
        "test/precision": precision,
        "test/recall": recall,
        "test/macro_f1": macro_f1,
    }

    return predictions, celltypes_labels, results, all_outputs




def evaluate_2(model: nn.Module, loader: DataLoader, device: torch.device, return_raw: bool = False) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_dab = 0.0
    total_num = 0
    predictions = []
    all_outputs = list()
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=(
                        batch_labels if INPUT_BATCH_LABELS or config.DSBN else None
                    ),
                    CLS=CLS,  # evaluation does not need CLS or CCE
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=do_sample_in_train,
                    # generative_training = False,
                )

                output_values = output_dict["cls_output"]
                loss = criterion_cls(output_values, celltype_labels)

                if DAB:
                    loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

            total_loss += loss.item() * len(input_gene_ids)
            accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
            total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
            total_dab += loss_dab.item() * len(input_gene_ids) if DAB else 0.0
            total_num += len(input_gene_ids)
            preds = output_values.argmax(1).cpu().numpy()
            predictions.append(preds)

            # convert everythin to cpu !
            output_dict = {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in output_dict.items()}
            
            all_outputs.append(output_dict)
            torch.cuda.empty_cache()

    wandb.log(
        {
            "valid/mse": total_loss / total_num,
            "valid/err": total_error / total_num,
            "valid/dab": total_dab / total_num,
            "valid/sum_mse_dab": (total_loss + dab_weight * total_dab) / total_num,
            "epoch": epoch,
        },
    )

    if return_raw:
        return np.concatenate(predictions, axis=0), all_outputs

    return total_loss / total_num, total_error / total_num, all_outputs


def generate_matrix(all_outputs: list[dict]) -> np.ndarray:
    """Generate Matrix from all_outputs dict
    Args:
        all_outputs (list[dict]): list of all_outputs dict
    Returns:
        cell_emb (np.ndarray): matrix of all cell embeddings
    """

    all_outputs = [o["cell_emb"] for o in all_outputs]
    all_outputs = torch.cat(all_outputs, dim=0)

    return all_outputs.cpu().numpy()

def log_cpu_memory_usage():
    memory = psutil.virtual_memory()
    logging.info(f"CPU Memory Usage: {memory.percent}% used of {memory.total / (1024 ** 3):.2f} GB total")


def log_gpu_memory_usage():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            cached = torch.cuda.memory_reserved(i) / (1024 ** 3)
            logging.info(f"GPU {i} Memory Usage: {allocated:.2f} GB allocated, {cached:.2f} GB cached")




def get_mask_genes(
    adata: AnnData,
    max_n_genes: int = 3501,
    gene_presence_pct: float = 0.95,
) -> np.array:
    """
    Filters genes based on presence and variance, returning a mask of selected genes.

    Parameters:
    - adata: AnnData object containing gene expression data
    - max_n_genes: The maximum number of genes to retain based on variance
    - gene_presence_pctl: Percentile of gene presence for the filtering threshold

    Returns:
    - A boolean array (mask) indicating which genes to keep.
    """
    total_n_genes = adata.X.shape[1]
    total_n_samples = adata.X.shape[0]

    # Step 1: Filter out based on presence
    count_presence = np.sum(~np.isnan(adata.X), axis=0)
    thr_presence = gene_presence_pct * total_n_samples
    mask_presence = count_presence > thr_presence
    logging.info(
        f"Presence threshold: {thr_presence:.0f} ({thr_presence/total_n_genes*100:.2f}%), {np.sum(mask_presence)} genes left"
    )

    # Step 2: Filter out based on variance
    count_variance = np.nanstd(adata.X, axis=0)  # Standard deviation (variance)

    # Step 3: Apply the presence mask and filter out NaN genes
    count_variance_masked = count_variance[mask_presence]
    count_variance_masked = count_variance_masked[~np.isnan(count_variance_masked)]

    # Step 4: Get the variance threshold, ensuring we don’t exceed available genes
    n_variance_masked_genes = len(count_variance_masked)
    thr_variance = np.sort(count_variance_masked)[::-1][
        min(max_n_genes, n_variance_masked_genes) - 1  # Adjust for 0-based indexing
    ]

    # Step 5: Create the variance mask
    mask_variance = count_variance >= thr_variance
    logging.info(
        f"Variance threshold: {thr_variance:.0f}, {np.sum(mask_variance)} genes left"
    )

    # Step 6: Combine presence and variance masks
    combined_mask = mask_presence & mask_variance
    logging.info(f"Combined mask: {np.sum(combined_mask)} genes left")

    return combined_mask


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

    # pre-process data
    obs_copy["combination"] = (
        obs_copy["celltype"].astype(str) + "_" + obs_copy["dataset_id"].astype(str)
    )

    logging.info(f"Columns in obs_copy {obs_copy.columns}")

    assert "combination" in obs_copy.columns, "combination column not created in obs"

    diseases_f1 = set()  # diseases filter 1

    # 1. Check diseases w/ 5+ datasets
    all_diseases = obs_copy["celltype"].unique()
    for diseases in all_diseases:
        QUERY = f'celltype == "{diseases}"'
        _df_query = obs_copy.query(QUERY)
        if len(_df_query["dataset_id"].unique()) >= 5:
            diseases_f1.add(diseases)

    logging.info(f"Number of diseases with 5+ datasets: {len(diseases_f1)}")

    # 2. Divide dataset into train and test w/ 4:1 ratio
    QUERY = "celltype in @diseases_f1"
    df_diseases_f1 = obs_copy.query(QUERY)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)
    for i, (train_idx, test_idx) in enumerate(
        sgkf.split(
            X=df_diseases_f1["ids"],
            y=df_diseases_f1["celltype"],
            groups=df_diseases_f1["dataset_id"],
        )
    ):

        # get which disease & datasets are in the test
        df_diseases_f1_test = df_diseases_f1.iloc[test_idx]

        assert "combination" in obs_copy.columns, "combination column not found in obs"
        assert (
            "combination" in df_diseases_f1_test.columns
        ), "combination column not found in df_diseases_f1_test"

        logging.info(
            f"Nº of diseases in test split {i+1}: {len(df_diseases_f1_test['celltype'].unique())}"
        )
        logging.info(
            f"Nº of datasets in test split {i+1}: {len(df_diseases_f1_test['dataset_id'].unique())}"
        )

        # 3. Assign train and test labels
        obs_copy[f"test_split_{i+1}"] = (
            obs_copy["combination"].isin(df_diseases_f1_test["combination"]).astype(int)
        )

        logging.info(
            f"Nº of samples in test split {i+1}: {obs_copy[f'test_split_{i+1}'].sum()}"
        )

    obs_copy.drop(columns=["combination"], inplace=True)

    return obs_copy

def get_old_test_split(adata: AnnData) -> np.array:
    """Get Old Test Split
    
    Args:
        adata (AnnData): AnnData object
    
    Returns:
        np.array: Binary array with 1s indicating test indices and 0s for train
    """
    import random

    # Combine disease type and dataset for shuffling to minimize bias
    labels = np.array(
        [a + b for a, b in zip(adata.obs["celltype"], adata.obs["dataset_id"])]
    )

    # Generate indices for the data points
    indices = np.arange(len(labels))

    # Find indices of labels that only occur once
    labels_count = {label: np.sum(labels == label) for label in np.unique(labels)}
    indices_single_label = [i for i, label in enumerate(labels) if labels_count[label] == 1]
    labels_single_label = labels[indices_single_label]

    logging.info(f"Nº of single label indexes: {len(indices_single_label)}")

    # Remaining indices (those with multiple occurrences)
    remaining_indices = np.setdiff1d(indices, indices_single_label)
    remaining_labels = labels[remaining_indices]

    # Perform stratified split on the remaining indices
    train_indices, test_indices = train_test_split(
        remaining_indices, test_size=0.2, stratify=remaining_labels, random_state=42
    )

    # Convert to lists for appending
    train_indices = train_indices.tolist()
    test_indices = test_indices.tolist()

    # Manually add single-label indices to train/test sets based on 80/20 split
    for idx in indices_single_label:
        if random.random() < 0.2:  # 20% chance to go to test set
            test_indices.append(idx)
        else:
            train_indices.append(idx)

    # Sort indices for consistency
    test_indices.sort()
    train_indices.sort()

    # Create binary array indicating test indices (1 = test, 0 = train)
    batch_ids = np.zeros(len(labels), dtype=int)
    batch_ids[test_indices] = 1  # Set test indices to 1

    logging.info(f"Nº of train samples: {np.sum(batch_ids == 0)}")
    logging.info(f"Nº of test samples: {np.sum(batch_ids == 1)}")

    return batch_ids.astype(bool)



def get_folder_name():
    # Step 1: Generate today's date string
    today = datetime.now().strftime("%y-%m-%d")

    # Step 2: Define the base output directory
    base_output_dir = os.path.join("..", "outputs")

    # Step 3: Find the highest existing run number for today
    existing_runs = [
        d for d in os.listdir(base_output_dir)
        if os.path.isdir(os.path.join(base_output_dir, d)) and d.startswith(f"run-{today}")
    ]

    # Extract numbers from existing runs and find the max
    existing_numbers = [
        int(d.split("-")[-1]) for d in existing_runs if d.split("-")[-1].isdigit()
    ]

    # Calculate the next run number
    next_run_number = max(existing_numbers, default=0) + 1

    # Step 4: Create the directory name with zero-padded run number
    output_dir = os.path.join(base_output_dir, f"run-{today}-{next_run_number:02d}")

    # Step 5: Create the directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory created: {output_dir}")
    return output_dir


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
            gene_variance = np.std(adata.X, axis=0)
            
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


#endregion

#region 1. Specify hyper-parameter setup for integration task

hyperparameter_defaults = dict(
    seed=0,
    dataset_name="test_1",
    do_train=True,
    # load_model="../save/scGPT_human",
    load_model = "/aloy/home/ddalton/projects/scGPT_playground/save/scGPT_human",
    mask_ratio=0.0,
    epochs=10,
    n_bins=51,
    MVC=False,  # Masked value prediction for cell embedding
    ecs_thres=0.0,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=0.0,
    lr=1e-4,
    batch_size=32,
    layer_size=128,
    nlayers=4,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead=4,  # number of heads in nn.MultiheadAttention
    dropout=0.2,  # dropout probability
    schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
    save_eval_interval=5,
    fast_transformer=True,
    pre_norm=False,
    amp=True,  # Automatic Mixed Precision
    include_zero_gene=False,
    freeze=False,  # freeze
    DSBN=False,  # Domain-spec batchnorm
)





# Load the config file
with open('/aloy/home/ddalton/projects/scGPT_playground/scripts/config/wandb.json', 'r') as f:
    api_config = json.load(f)

# Use the Wandb API key from the config file
if 'wandb_api_key' in api_config:
    os.environ['WANDB_API_KEY'] = api_config['wandb_api_key']

# Log in to Wandb using the API key
wandb.login()

run = wandb.init(
    config=hyperparameter_defaults,
    project="scGPT",
    reinit=True,
    settings=wandb.Settings(start_method="fork"),
)
config = wandb.config
print(config)

set_seed(config.seed)





# settings for input and preprocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = config.mask_ratio
mask_value = "auto"  # for masked values, now it should always be auto

include_zero_gene = (
    config.include_zero_gene
)  # if True, include zero genes among hvgs in the training

# max_seq_len = 9062  # adata.X.shape[1]+1
n_bins = config.n_bins

# input/output representation
input_style = "binned"  # "normed_raw", "log1p", or "binned"
output_style = "binned"  # "normed_raw", "log1p", or "binned"

# settings for training
MLM = False  # whether to use masked language modeling, currently it is always on.
CLS = True  # celltype classification objective
ADV = False  # Adversarial training for batch correction
CCE = False  # Contrastive cell embedding objective
MVC = config.MVC  # Masked value prediction for cell embedding
ECS = config.ecs_thres > 0  # Elastic cell similarity objective
DAB = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
INPUT_BATCH_LABELS = False  # TODO: have these help MLM and MVC, while not to classifier
input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"
adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
adv_D_delay_epochs = 0
mvc_decoder_style = "inner product"
ecs_threshold = config.ecs_thres
dab_weight = config.dab_weight

explicit_zero_prob = MLM and include_zero_gene  # whether explicit bernoulli for zeros
do_sample_in_train = False and explicit_zero_prob  # sample the bernoulli in training

per_seq_batch_sample = False

# settings for optimizer
lr = config.lr  # TODO: test learning rate ratio between two tasks
lr_ADV = 1e-3  # learning rate for discriminator, used when ADV is True

#! CHANGED TRANSFORMER BLOCK
#! changing batch_size
batch_size = manual_parameters.get("batch_size")
eval_batch_size = batch_size
# batch_size = config.batch_size
# eval_batch_size = config.batch_size

epochs = config.epochs
schedule_interval = 1

# settings for the model
fast_transformer = config.fast_transformer
fast_transformer_backend = "flash"  # "linear" or "flash"
embsize = config.layer_size  # embedding dimension
d_hid = config.layer_size  # dimension of the feedforward network in TransformerEncoder
nlayers = config.nlayers  # number of TransformerEncoderLayer in TransformerEncoder
nhead = config.nhead  # number of heads in nn.MultiheadAttention
dropout = config.dropout  # dropout probability

# logging
log_interval = 100  # iterations
save_eval_interval = config.save_eval_interval  # epochs
do_eval_scib_metrics = True


assert input_style in ["normed_raw", "log1p", "binned"]
assert output_style in ["normed_raw", "log1p", "binned"]
assert input_emb_style in ["category", "continuous", "scaling"]
if input_style == "binned":
    if input_emb_style == "scaling":
        raise ValueError("input_emb_style `scaling` is not supported for binned input.")
elif input_style == "log1p" or input_style == "normed_raw":
    if input_emb_style == "category":
        raise ValueError(
            "input_emb_style `category` is not supported for log1p or normed_raw input."
        )

if input_emb_style == "category":
    mask_value = n_bins + 1
    pad_value = n_bins  # for padding gene expr values
    n_input_bins = n_bins + 2
else:
    mask_value = -1
    pad_value = -2
    n_input_bins = n_bins

if ADV and DAB:
    raise ValueError("ADV and DAB cannot be both True.")
DAB_separate_optim = True if DAB > 1 else False


dataset_name = config.dataset_name
save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")

#endregion

#region 2. Load and pre-process data

# We follow the standard scGPT data pre-processing pipelines for the cell-type annotation task. Note that since now we have two datasets at hand (i.e., reference and query data), the same pre-prpocessing steps need to be applied to both of them.


"""adata information required

.obs["celltype"]
.obs["batch_id"]
.var["gene_name"]
.obs["celltype_id"]

"""

if False:
    if dataset_name == "ms":
        data_dir = Path("../data/ms")
        adata = sc.read(data_dir / "c_data.h5ad")
        adata_test = sc.read(data_dir / "filtered_ms_adata.h5ad")
        adata.obs["celltype"] = adata.obs[
            "Factor Value[inferred cell type - authors labels]"
        ].astype("category")
        adata_test.obs["celltype"] = adata_test.obs[
            "Factor Value[inferred cell type - authors labels]"
        ].astype("category")
        adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
        adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"
        adata.var.set_index(adata.var["gene_name"], inplace=True)
        adata_test.var.set_index(adata.var["gene_name"], inplace=True)
        data_is_raw = False
        filter_gene_by_counts = False
        adata_test_raw = adata_test.copy()
        adata = adata.concatenate(adata_test, batch_key="str_batch")

    # make the batch category column
    batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
    adata.obs["batch_id"] = batch_id_labels
    celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
    celltypes = adata.obs["celltype"].unique()
    num_types = len(np.unique(celltype_id_labels))
    id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
    adata.obs["celltype_id"] = celltype_id_labels
    adata.var["gene_name"] = adata.var.index.tolist()


# data_dir = Path("../data/test_1/")
# adata = sc.read(data_dir / "test_2.h5ad")
adata = sc.read(manual_parameters.get("data_path"))


# config parameters
data_is_raw = True
filter_gene_by_counts = False

# make the batch category column

celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
celltypes = adata.obs["celltype"].unique()
num_types = len(np.unique(celltype_id_labels))
id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
adata.obs["celltype_id"] = celltype_id_labels



# presence = 0.95  # gene presence in samples
# variance = 45  # gene variance in top %

# # filter out the genes with lowest presence
# thr_presence = adata.X.shape[0] * presence

# mask_presence = np.sum(~np.isnan(adata.X), axis=0) > thr_presence

# logging.info(f"Presence thr {thr_presence}, {np.sum(mask_presence)} genes left")


# # filter out the genes with lowest standard deviation
# gene_std = np.nanstd(adata.X, axis=0)
# gene_std = gene_std[~np.isnan(gene_std)]
# thr_std = np.percentile(gene_std, variance)

# mask_variance = np.nanstd(adata.X, axis=0) > thr_std


# logging.info(f"Variance thr {thr_std:.2f}, {np.sum(mask_variance)} genes left")

# # combine mask
# mask_genes = mask_presence & mask_variance


# mask_genes = get_mask_genes(adata, 
#                   max_n_genes=manual_parameters.get("max_seq_len"), 
#                   gene_presence_pct=manual_parameters.get("gene_presence_pct"))



mask_genes = get_top_k_most_present_genes(adata, 
                  k=manual_parameters.get("max_seq_len"))

logging.info(f"Combined mask {np.sum(mask_genes)} genes left")

adata_orig = adata.copy()

# mask the genes
adata = adata[:, mask_genes]


# mask samples
non_nan_percentage = np.sum(~np.isnan(adata.X), axis=1) / adata.X.shape[1]

# mask samples that have less than 30% non-NaN values
mask_samples = non_nan_percentage >= 0.30
logging.info(f"Masking {np.sum(~mask_samples)} samples with less than 30% non-NaN values")

# apply the mask to the AnnData object
adata = adata[mask_samples, :]


#! COMMENTED
# adata.obs["str_batch"] = adata.obs["train_test"].astype(int).astype(str)

if config.load_model is not None:
    model_dir = Path(config.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    shutil.copy(vocab_file, save_dir / "vocab.json")
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    # model
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(
        f"Resume model from {model_file}, the model args will override the "
        f"config {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]


# set up the preprocessor, use the args to config the workflow
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


# adata_test = adata[adata.obs["str_batch"] == "1"]
# adata = adata[adata.obs["str_batch"] == "0"]

#! CHANGED
#! TEST SPLIT FUNCTION
if manual_parameters.get("benchmark_data"):

    adata_test = adata[adata.obs["str_batch"] == "1"]
    adata = adata[adata.obs["str_batch"] == "0"]

    # added
    adata_test_raw = adata_test.copy()


else:
    if manual_parameters.get("new_split"):
        df_obs = adata.obs
        new_obs = get_test_split(obs=df_obs,n_splits=5)
        adata.obs = new_obs
        adata_test = adata[adata.obs["test_split_3"] == 1]
        adata = adata[adata.obs["test_split_3"] == 0]
        adata.obs["str_batch"] = adata.obs["test_split_1"].astype(int).astype(str)
    else:
        mask_old_split = get_old_test_split(adata)
        adata_test = adata[mask_old_split]
        adata = adata[~mask_old_split]

# sys.exit(0)

# added
adata_test_raw = adata_test.copy()

preprocessor(adata, batch_key=None)
preprocessor(adata_test, batch_key=None)


input_layer_key = {  # the values of this map coorespond to the keys in preprocessing
    "normed_raw": "X_normed",
    "log1p": "X_normed",
    "binned": "X_binned",
}[input_style]
all_counts = (
    adata.layers[input_layer_key].A
    if issparse(adata.layers[input_layer_key])
    else adata.layers[input_layer_key]
)
genes = adata.var["gene_name"].tolist()

celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
celltypes_labels = np.array(celltypes_labels)

batch_ids = adata.obs["batch_id"].tolist()
num_batch_types = len(set(batch_ids))
batch_ids = np.array(batch_ids)

(
    train_data,
    valid_data,
    train_celltype_labels,
    valid_celltype_labels,
    train_batch_labels,
    valid_batch_labels,
) = train_test_split(
    all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
)

Counter(celltypes_labels)


if config.load_model is None:
    vocab = Vocab(
        VocabPybind(genes + special_tokens, None)
    )  # bidirectional lookup [gene <-> int]
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)


tokenized_train = tokenize_and_pad_batch(
    train_data,
    gene_ids,
    max_len=manual_parameters.get("max_seq_len"),
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,  # append <cls> token at the beginning
    include_zero_gene=include_zero_gene,
)
tokenized_valid = tokenize_and_pad_batch(
    valid_data,
    gene_ids,
    max_len=manual_parameters.get("max_seq_len"),
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=include_zero_gene,
)
logger.info(
    f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
)
logger.info(
    f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
)

# endregion

#region 3. Load the pre-trained scGPT model


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ntokens = len(vocab)  # size of vocabulary
model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=3,
    n_cls=num_types if CLS else 1,
    vocab=vocab,
    dropout=dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=MVC,
    do_dab=DAB,
    use_batch_labels=INPUT_BATCH_LABELS,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=config.DSBN,
    input_emb_style=input_emb_style,
    n_input_bins=n_input_bins,
    cell_emb_style=cell_emb_style,
    mvc_decoder_style=mvc_decoder_style,
    ecs_threshold=ecs_threshold,
    explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=fast_transformer,
    fast_transformer_backend=fast_transformer_backend,
    pre_norm=config.pre_norm,
)
if config.load_model is not None:
    try:
        model.load_state_dict(torch.load(model_file))
        logger.info(f"Loading all model params from {model_file}")
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
            # logger.info(f"Loading params {k} with shape {v.shape}")
            pass
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

pre_freeze_param_count = sum(
    dict(
        (p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad
    ).values()
)

# Freeze all pre-decoder weights
for name, para in model.named_parameters():
    # print("-" * 20)
    # print(f"name: {name}")
    if config.freeze and "encoder" in name and "transformer_encoder" not in name:
        # if config.freeze and "encoder" in name:
        # print(f"freezing weights for: {name}")
        para.requires_grad = False

post_freeze_param_count = sum(
    dict(
        (p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad
    ).values()
)

logger.info(f"Total Pre freeze Params {(pre_freeze_param_count )}")
logger.info(f"Total Post freeze Params {(post_freeze_param_count )}")
wandb.log(
    {
        "info/pre_freeze_param_count": pre_freeze_param_count,
        "info/post_freeze_param_count": post_freeze_param_count,
    },
)

model.to(device)
wandb.watch(model)

if ADV:
    discriminator = AdversarialDiscriminator(
        d_model=embsize,
        n_cls=num_batch_types,
    ).to(device)


criterion = masked_mse_loss
criterion_cls = nn.CrossEntropyLoss()
criterion_dab = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=lr, eps=1e-4 if config.amp else 1e-8
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, schedule_interval, gamma=config.schedule_ratio
)
if DAB_separate_optim:
    optimizer_dab = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler_dab = torch.optim.lr_scheduler.StepLR(
        optimizer_dab, schedule_interval, gamma=config.schedule_ratio
    )
if ADV:
    criterion_adv = nn.CrossEntropyLoss()  # consider using label smoothing
    optimizer_E = torch.optim.Adam(model.parameters(), lr=lr_ADV)
    scheduler_E = torch.optim.lr_scheduler.StepLR(
        optimizer_E, schedule_interval, gamma=config.schedule_ratio
    )
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_ADV)
    scheduler_D = torch.optim.lr_scheduler.StepLR(
        optimizer_D, schedule_interval, gamma=config.schedule_ratio
    )

scaler = torch.cuda.amp.GradScaler(enabled=config.amp)


#endregion




#region 4. Finetune scGPT with task-specific objectives

if torch.cuda.is_available():
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    torch.cuda.empty_cache()


else:
    print("CUDA is not available")


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"



best_val_loss = float("inf")
best_avg_bio = 0.0
best_model = None
define_wandb_metrcis()

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    # train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample)
    train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=False)
    train_loader = prepare_dataloader(
        train_data_pt,
        batch_size=batch_size,
        shuffle=False,
        intra_domain_shuffle=True,
        drop_last=False,
    )
    valid_loader = prepare_dataloader(
        valid_data_pt,
        batch_size=eval_batch_size,
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
    )

    if config.do_train:
        train(
            model,
            loader=train_loader,
        )
    val_loss, val_err = evaluate(
        model,
        loader=valid_loader,
    )
    elapsed = time.time() - epoch_start_time
    logger.info("-" * 89)
    logger.info(
        f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
        f"valid loss/mse {val_loss:5.4f} | err {val_err:5.4f}"
    )
    logger.info("-" * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        best_model_epoch = epoch
        logger.info(f"Best model with score {best_val_loss:5.4f}")

    scheduler.step()
    if DAB_separate_optim:
        scheduler_dab.step()
    if ADV:
        scheduler_D.step()
        scheduler_E.step()

    #! FOR DEBUGGING
    # break

val_loss, val_err = evaluate(
    model,
    loader=valid_loader,
)


# ## Step 5: Inference with fine-tuned scGPT model
# In the cell-type annotation task, the fine-tuned scGPT predicts cell-type labels for query set as inference. The model performance is evaluated on standard classificaton metrics. Here we visualize the predicted labels over the scGPT cell embeddings, and present the confusion matrix for detailed classification performance on the cell-group level.


predictions, labels, results = test(best_model, adata_test)


print(Counter(labels), Counter(predictions))


# get rid of nans
adata_test_raw.X = np.where(np.isnan(adata_test_raw.X), 0, adata_test_raw.X)


# Check if PCA is computed; if not, compute it
if "X_pca" not in adata_test_raw.obsm:
    sc.pp.pca(adata_test_raw)

# Compute neighbors and UMAP
if "X_umap" not in adata_test_raw.obsm:
    sc.pp.neighbors(
        adata_test_raw, n_neighbors=15, use_rep="X_pca"
    )  # Adjust parameters as needed
    sc.tl.umap(adata_test_raw)





adata_test_raw.obs["predictions"] = [id2type[p] for p in predictions]

# plot
palette_ = plt.rcParams["axes.prop_cycle"].by_key()["color"]
palette_ = (
    plt.rcParams["axes.prop_cycle"].by_key()["color"]
    + plt.rcParams["axes.prop_cycle"].by_key()["color"]
    + plt.rcParams["axes.prop_cycle"].by_key()["color"]
)
palette_ = {c: palette_[i] for i, c in enumerate(celltypes)}

with plt.rc_context({"figure.figsize": (6, 6), "figure.dpi": (300)}):
    sc.pl.umap(
        adata_test_raw, color=["celltype", "predictions"], palette=palette_, show=False
    )
    plt.savefig(save_dir / "results.png", dpi=300)

save_dict = {
    "predictions": predictions,
    "labels": labels,
    "results": results,
    "id_maps": id2type,
}
with open(save_dir / "results.pkl", "wb") as f:
    pickle.dump(save_dict, f)

results["test/cell_umap"] = wandb.Image(
    str(save_dir / "results.png"),
    caption=f"predictions macro f1 {results['test/macro_f1']:.3f}",
)
wandb.log(results)


celltypes = list(celltypes)
for i in set([id2type[p] for p in predictions]):
    if i not in celltypes:
        celltypes.remove(i)

print(len(labels), len(predictions))
cm = confusion_matrix(labels, predictions)
cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

# sorted celltypes by the order of the confusion matrix
sorted_celltypes = list()
for i in sorted(list(set(labels))):
    sorted_celltypes.append(id2type[i])

cm = pd.DataFrame(
    cm, index=sorted_celltypes[: cm.shape[0]], columns=sorted_celltypes[: cm.shape[1]]
)

plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues")
plt.savefig(save_dir / "confusion_matrix.png", dpi=300)

results["test/confusion_matrix"] = wandb.Image(
    str(save_dir / "confusion_matrix.png"),
    caption=f"confusion matrix",
)



# save the model into the save_dir
torch.save(best_model.state_dict(), save_dir / "model.pt")

## Evaluate the model on the train set

predictions_train, labels_train, results_train = test(best_model, adata)


# ### Evaluate results on train

logging.info("region 4")
log_cpu_memory_usage()
log_gpu_memory_usage()

#endregion

#region 5. Inference with fine-tuned scGPT model
adata_train_raw = adata.copy()


# get predictions
# test inference
(
    predictions_test,
    labels_test,
    results_test,
    all_outputs_test,
) = test_2(best_model, adata_test)

logging.info(f"Results Test: {results_test}")



# train inference
(
    predictions_train,
    labels_train,
    results_train,
    all_outputs_train,
) = test_2(best_model, adata)




# if available_device is not None:
#     logging.info(f"Switching model to available device found: {available_device}")

#     best_model.to(available_device)

# else:
#     logging.info("No available device found, using the default device")
#     sys.exit(1)




logging.info("region 5")
log_cpu_memory_usage()
log_gpu_memory_usage()

#endregion



#region 6. Save output

# Generate the output directory
output_dir = get_folder_name()

# Prepare a dictionary of all variables to be saved
data_to_save = {
    "predictions_test": predictions_test,
    "labels_test": labels_test,
    "results_test": results_test,
    "all_outputs_test": all_outputs_test,
    "predictions_train": predictions_train,
    "labels_train": labels_train,
    "results_train": results_train,
    "all_outputs_train": all_outputs_train,
    "adata_train": adata,
    "adata_test": adata_test,
    "id2type": id2type,
}

# Save each item in the dictionary to a pickle file
for filename, data in data_to_save.items():
    if filename.startswith("adata"):
        file_path = os.path.join(output_dir, f"{filename}.h5ad")
        data.write(file_path)
    else:
        file_path = os.path.join(output_dir, f"{filename}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
            
            
logging.info("region 4")
log_cpu_memory_usage()
log_gpu_memory_usage()

# save manual parameters
# Write parameters to a JSON file
with open(os.path.join(output_dir,"parameters.json"), 'w') as json_file:
    json.dump(manual_parameters, json_file, indent=4)


#! ORIGINAL CODE-BLOCKS
# #region 5. Inference with fine-tuned scGPT model

# adata_train_raw = adata.copy()

# # get predictions
# predictions, labels, results = test(best_model, adata_train_raw)


# # get rid of nans
# adata_train_raw.X = np.where(np.isnan(adata_train_raw.X), 0, adata_train_raw.X)


# logging.info(f"adata train shape {adata_train_raw.X.shape}")



# # Check if PCA is computed; if not, compute it
# if "X_pca" not in adata_train_raw.obsm:
#     sc.pp.pca(adata_train_raw)

# # Compute neighbors and UMAP
# if "X_umap" not in adata_train_raw.obsm:
#     sc.pp.neighbors(
#         adata_train_raw, n_neighbors=15, use_rep="X_pca"
#     )  # Adjust parameters as needed
#     sc.tl.umap(adata_train_raw)
# adata_train_raw.obs["predictions"] = [id2type[p] for p in predictions]





# # plot
# palette_ = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# palette_ = (
#     plt.rcParams["axes.prop_cycle"].by_key()["color"]
#     + plt.rcParams["axes.prop_cycle"].by_key()["color"]
#     + plt.rcParams["axes.prop_cycle"].by_key()["color"]
# )
# palette_ = {c: palette_[i] for i, c in enumerate(celltypes)}

# with plt.rc_context({"figure.figsize": (6, 4), "figure.dpi": (300)}):
#     sc.pl.umap(
#         adata_train_raw,
#         color=["celltype", "predictions"],
#         palette=palette_,
#         show=False,
#     )
#     plt.savefig(save_dir / "results.train.png", dpi=300)

# save_dict = {
#     "predictions": predictions,
#     "labels": labels,
#     "results": results,
#     "id_maps": id2type,
# }
# with open(save_dir / "results.pkl", "wb") as f:
#     pickle.dump(save_dict, f)

# results["test/cell_umap"] = wandb.Image(
#     str(save_dir / "results.png"),
#     caption=f"predictions macro f1 {results['test/macro_f1']:.3f}",
# )
# wandb.log(results)


# celltypes = list(celltypes)
# for i in set([id2type[p] for p in predictions]):
#     if i not in celltypes:
#         celltypes.remove(i)

# print(len(labels), len(predictions))

# cm = confusion_matrix(labels, predictions)
# cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

# # sorted celltypes by the order of the confusion matrix
# sorted_celltypes = list()
# for i in sorted(list(set(labels))):
#     sorted_celltypes.append(id2type[i])

# cm = pd.DataFrame(
#     cm, index=sorted_celltypes[: cm.shape[0]], columns=sorted_celltypes[: cm.shape[1]]
# )
# plt.figure(figsize=(10, 10))
# sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues")
# plt.savefig(save_dir / "confusion_matrix.png", dpi=300)

# results["test/confusion_matrix"] = wandb.Image(
#     str(save_dir / "confusion_matrix.png"),
#     caption=f"confusion matrix",
# )


# scg.tasks.embed_data(adata_train_raw, "save/dev_test_1-Jul30-16-04/", gene_col="index")

#endregion

# # ## Testing Area


# (
#     predictions_2,
#     labels_2,
#     results_2,
#     all_outputs,
# ) = test_2(best_model, adata_test)


# print(results_2)


# np.concatenate(
#     (all_outputs[0]["cell_emb"].cpu(), all_outputs[0]["cell_emb"].cpu()), axis=0
# ).shape




# cell_emb = generate_matrix(all_outputs)



# # umap plot of cell embeddings
# import umap
# import matplotlib.pyplot as plt

# list_labels = adata_train_raw.obs["celltype"]

# labels = list(set(list_labels))

# irb_colors = [
#     "#ffd81cff",
#     "#f6972dff",
#     "#f2612dff",
#     "#574270ff",
#     "#00589bff",
#     "#002f58ff",
# ]

# # Create UMAP embeddings
# reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
# X_test_umap = reducer.fit_transform(cell_emb)

# color_map = {label: color for label, color in zip(labels, irb_colors)}
# plt.savefig(save_dir / "umap_truelabels.png", dpi=300)
# plt.close()





# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(figsize=(10, 5))

# # UMAP colored by true labels
# for label in labels:
#     idx = list_labels == label
#     ax.scatter(
#         X_test_umap[idx, 0],
#         X_test_umap[idx, 1],
#         c=color_map[label],
#         label=label,
#         s=35,
#         alpha=0.5,
#     )
# ax.set_title("UMAP - scGPT embeddings Colored by True Labels")
# ax.legend()
# ax.set_xlabel("UMAP 1")
# ax.set_ylabel("UMAP 2")

# plt.savefig(save_dir / "umap_true_labels.png", dpi=300)

