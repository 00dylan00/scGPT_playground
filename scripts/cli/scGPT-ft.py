"""Tutorial Annotation

Structure:
    1. Specify hyper-parameter setup for integration task
    2. Load and pre-process data
    3. Load the pre-trained scGPT model
    4. Finetune scGPT with task-specific objectives
    5. Inference with fine-tuned scGPT model
    6. Save output
"""

# region 0. Imports, Variables & Functions

import copy, json, os
from pathlib import Path
import shutil, sys, time
from typing import *
import warnings, pandas as pd
from datetime import datetime
import sys

sys.path.append("/aloy/home/ddalton/projects/scGPT_playground/")
from scanpy.pp import combat
from sklearn.metrics import average_precision_score, roc_auc_score

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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, "../")
sys.path.insert(0, "/aloy/home/ddalton/git_clones/scGPT")
import scgpt as scg

print(f"scGPT version: {scg.__file__}")

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
from src.training import helpers as tr_h

from collections import Counter
import logging
import psutil

from typing import *
from sklearn.model_selection import StratifiedGroupKFold, KFold
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings("ignore")

logging.info(f"Is Cuda Available {str(torch.cuda.is_available())}")


import torch

print(torch.cuda.device_count())  # Check how many GPUs are available
parser = argparse.ArgumentParser(description="Script for scGPT project")


# Define arguments
parser.add_argument(
    "--data_path", type=str, required=True, help="Path to the data file"
)
parser.add_argument(
    "--max_seq_len", type=int, default=1000, help="Maximum sequence length"
)
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument(
    "--gene_presence_pct", type=float, default=0.9, help="Gene presence percentage"
)
parser.add_argument(
    "--split_type", type=str, default="stratified", help="Type of gene filtering"
)
parser.add_argument(
    "--val_split_type", type=str, default="random", help="Type of gene filtering"
)
parser.add_argument(
    "--n_splits", type=int, default=10, help="Number of splits for cross-validation"
)  # old parameter - now refers to split ratio 1:9 test in case of 10!
parser.add_argument(
    "--n_tested_splits", type=int, default=None, help="Number of tested splits"
)
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument(
    "--gene_filtering", type=str, default="top_presence", help="Type of gene filtering"
)
parser.add_argument("--ecs_thres", type=float, default=0.0, help="Include ecs_thres")
parser.add_argument("--dab_weight", type=float, default=0.0, help="Include dab_weight")
parser.add_argument("--ontology", type=str, default="do", help="Type of ontology")

# booleans can be a pain in the ass with parser
parser.add_argument(
    "--benchmark_data", type=eval, default=False, help="Use benchmark data"
)
parser.add_argument("--MLM", type=eval, default=False, help="Include MLM")
parser.add_argument("--CLS", type=eval, default=True, help="Include CLS")
parser.add_argument(
    "--CLS_multilabel", type=eval, default=True, help="Include CLS for multilabel"
)
parser.add_argument("--CCE", type=eval, default=False, help="Include CCE")
parser.add_argument(
    "--DAB", type=eval, default=False, help="Domain Adversarial Back Propagation"
)
parser.add_argument(
    "--ADV", type=eval, default=False, help="Include Adversarial Training"
)
parser.add_argument(
    "--use_fast_transformer", type=eval, default=True, help="Use fast transformer"
)
parser.add_argument(
    "--output_attentions", type=eval, default=False, help="Output attention scores"
)
parser.add_argument(
    "--INPUT_BATCH_LABELS",
    type=eval,
    default=False,
    help="Use batch labels in model input",
)
parser.add_argument(
    "--do_combat", type=eval, default=False, help="Apply ComBat batch correction"
)


# Parse arguments
args = parser.parse_args()
print("######\tParsed arguments\t######")
print(args)


if args.n_tested_splits is None:
    args.n_tested_splits = args.n_splits


# Use the arguments in your script
print(f"Data path: {args.data_path}")
print(f"Batch size: {args.batch_size}")

# variables
# data_path = "../data/test_1/test_2.h5ad"
manual_parameters = {
    "data_path": args.data_path,
    "max_seq_len": args.max_seq_len,
    "batch_size": args.batch_size,
    "gene_presence_pct": args.gene_presence_pct,
    "benchmark_data": args.benchmark_data,
    "split_type": args.split_type,
    "val_split_type": args.val_split_type,
    "n_splits": args.n_splits,
    "n_tested_splits": args.n_tested_splits,
    "epochs": args.epochs,
    "gene_filtering": args.gene_filtering,
    "sample_presence_pct": 0.3,
    "MLM": args.MLM,
    "CLS": args.CLS,
    "CLS_multilabel": args.CLS_multilabel,
    "DAB": args.DAB,
    "ADV": args.ADV,
    "CCE": args.CCE,
    "ecs_thres": args.ecs_thres,
    "dab_weight": args.dab_weight,
    "use_fast_transformer": args.use_fast_transformer,
    "output_attentions": args.output_attentions,
    "INPUT_BATCH_LABELS": args.INPUT_BATCH_LABELS,
    "do_combat": args.do_combat,
    "ontology": args.ontology,
}


for k, v in manual_parameters.items():
    print(f"{k}: {v}")


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

    _loss = list()
    _error_rate = list()
    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        celltype_labels = batch_data["celltype_labels"].to(device)
        if CLS_MULTILABEL:
            disease_multilabel = batch_data["class_multilabel"].to(device).float()

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        with torch.cuda.amp.autocast(enabled=config.amp):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=(
                    batch_labels if INPUT_BATCH_LABELS or config.DSBN else None
                ),
                CLS=(CLS or CLS_MULTILABEL),
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

            if CLS_MULTILABEL:
                loss_cls_multilabel = criterion_cls_multilabel(
                    output_dict["cls_output"], disease_multilabel
                )
                loss = loss + loss_cls_multilabel
                metrics_to_log.update(
                    {"train/cls_multilabel": loss_cls_multilabel.item()}
                )

                # error rate is 1 - accuracy
                _preds = torch.sigmoid(output_dict["cls_output"].detach()).cpu().numpy()
                _preds_bin = (_preds > 0.5).astype(int)
                _true = disease_multilabel.cpu().numpy()
                samplewise_acc = (_preds_bin == _true).mean(axis=1).mean()
                error_rate = 1 - samplewise_acc

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

        # clears the previous gradients from the model's parameters
        # necessary bc PyTorch accumulates gradients by default
        model.zero_grad()

        # the actual backpropagation
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
                CLS=(CLS or CLS_MULTILABEL),
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

        _loss.append(loss.item())
        _error_rate.append(1 - error_rate)

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

    return np.mean(_loss), np.mean(_error_rate)


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

            if CLS_MULTILABEL:
                disease_multilabel = batch_data["class_multilabel"].to(device).float()

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=(
                        batch_labels if INPUT_BATCH_LABELS or config.DSBN else None
                    ),
                    CLS=(CLS or CLS_MULTILABEL),  # evaluation does not need CLS or CCE
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=do_sample_in_train,
                    # generative_training = False,
                )

                if CLS:
                    output_values = output_dict["cls_output"]
                    loss = criterion_cls(output_values, celltype_labels)
                if CLS_MULTILABEL:
                    output_values = output_dict["cls_output"]
                    loss = criterion_cls_multilabel(output_values, disease_multilabel)

                if DAB:
                    loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

            total_loss += loss.item() * len(input_gene_ids)
            total_dab += loss_dab.item() * len(input_gene_ids) if DAB else 0.0
            total_num += len(input_gene_ids)

            if CLS:
                # Standard single-label accuracy
                accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
                total_error += (1 - accuracy / len(input_gene_ids)) * len(
                    input_gene_ids
                )
                preds = output_values.argmax(1).cpu().numpy()

            elif CLS_MULTILABEL:
                # Multilabel: compute sigmoid + threshold
                preds = torch.sigmoid(output_values).cpu().numpy()
                preds_bin = (preds > 0.5).astype(int)
                true = disease_multilabel.cpu().numpy()

                # Sample-wise accuracy: proportion of labels correctly predicted
                samplewise_acc = (preds_bin == true).mean(axis=1).mean()
                total_error += (1 - samplewise_acc) * len(input_gene_ids)

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

    if CLS_MULTILABEL:
        d_y_multilabel = u.get_multilabel_dict_from_adata(adata)
        disease_multilabels = np.array(d_y_multilabel["Y_multilabel"])

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
    if CLS_MULTILABEL:
        test_data_pt["class_multilabel"] = torch.from_numpy(disease_multilabels).float()

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

    if CLS:
        # compute accuracy, precision, recall, f1
        accuracy = accuracy_score(celltypes_labels, predictions)
        precision = precision_score(celltypes_labels, predictions, average="macro")
        recall = recall_score(celltypes_labels, predictions, average="macro")
        macro_f1 = f1_score(celltypes_labels, predictions, average="macro")

    elif CLS_MULTILABEL:
        # compute accuracy, precision, recall, f1
        predictions_bin = (predictions > 0.5).astype(int)
        print("predictions_bin", predictions_bin)
        print("disease_multilabels", disease_multilabels)
        accuracy = accuracy_score(disease_multilabels, predictions_bin)
        precision = precision_score(
            disease_multilabels, predictions_bin, average="macro"
        )
        recall = recall_score(disease_multilabels, predictions_bin, average="macro")
        macro_f1 = f1_score(disease_multilabels, predictions_bin, average="macro")

        # computer average precision & auroc
        avg_precision = average_precision_score(
            disease_multilabels, predictions, average="macro"
        )
        auroc = roc_auc_score(
            disease_multilabels, predictions, average="micro"
        )  #! CHANGED BC SOMETIMES COMPLAINS
        print(f"Average Precision: {avg_precision:.3f}, AUROC: {auroc:.3f}")

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

    if CLS_MULTILABEL:
        d_y_multilabel = u.get_multilabel_dict_from_adata(adata)
        disease_multilabels = np.array(d_y_multilabel["Y_multilabel"])

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
    if CLS_MULTILABEL:
        test_data_pt["class_multilabel"] = torch.from_numpy(disease_multilabels).float()

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

    if CLS:
        # compute accuracy, precision, recall, f1
        accuracy = accuracy_score(celltypes_labels, predictions)
        precision = precision_score(celltypes_labels, predictions, average="macro")
        recall = recall_score(celltypes_labels, predictions, average="macro")
        macro_f1 = f1_score(celltypes_labels, predictions, average="macro")

    elif CLS_MULTILABEL:
        # compute accuracy, precision, recall, f1
        predictions_bin = (predictions > 0.5).astype(int)
        accuracy = accuracy_score(disease_multilabels, predictions_bin)
        precision = precision_score(
            disease_multilabels, predictions_bin, average="macro"
        )
        recall = recall_score(disease_multilabels, predictions_bin, average="macro")
        macro_f1 = f1_score(disease_multilabels, predictions_bin, average="macro")

        # computer average precision & auroc
        from sklearn.metrics import average_precision_score, roc_auc_score

        avg_precision = average_precision_score(
            disease_multilabels, predictions, average="macro"
        )
        auroc = roc_auc_score(
            disease_multilabels, predictions, average="micro"
        )  #! CHANGED BC SOMETIMES COMPLAINS
        print(f"Average Precision: {avg_precision:.3f}, AUROC: {auroc:.3f}")

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


def evaluate_2(
    model: nn.Module, loader: DataLoader, device: torch.device, return_raw: bool = False
) -> float:
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

            if CLS_MULTILABEL:
                disease_multilabel = batch_data["class_multilabel"].to(device).float()

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=(
                        batch_labels if INPUT_BATCH_LABELS or config.DSBN else None
                    ),
                    CLS=(CLS or CLS_MULTILABEL),  # evaluation does not need CLS or CCE
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=do_sample_in_train,
                    # generative_training = False,
                )

                if CLS:
                    output_values = output_dict["cls_output"]
                    loss = criterion_cls(output_values, celltype_labels)
                if CLS_MULTILABEL:
                    output_values = output_dict["cls_output"]
                    loss = criterion_cls_multilabel(output_values, disease_multilabel)

                if DAB:
                    loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

            total_loss += loss.item() * len(input_gene_ids)
            total_dab += loss_dab.item() * len(input_gene_ids) if DAB else 0.0
            total_num += len(input_gene_ids)

            total_loss += loss.item() * len(input_gene_ids)
            total_dab += loss_dab.item() * len(input_gene_ids) if DAB else 0.0
            total_num += len(input_gene_ids)

            if CLS:
                # Standard single-label accuracy
                accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
                total_error += (1 - accuracy / len(input_gene_ids)) * len(
                    input_gene_ids
                )
                preds = output_values.argmax(1).cpu().numpy()

            elif CLS_MULTILABEL:
                # Multilabel: compute sigmoid + threshold
                preds = torch.sigmoid(output_values).cpu().numpy()
                preds_bin = (preds > 0.5).astype(int)
                true = disease_multilabel.cpu().numpy()

                # Sample-wise accuracy: proportion of labels correctly predicted
                samplewise_acc = (preds_bin == true).mean(axis=1).mean()
                total_error += (1 - samplewise_acc) * len(input_gene_ids)

            predictions.append(preds)

            # convert everythin to cpu !
            output_dict = {
                key: value.cpu() if isinstance(value, torch.Tensor) else value
                for key, value in output_dict.items()
            }

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

    if CLS_MULTILABEL:
        # 👇👇👇 Add this for multilabel
        tensor_class_multilabel_train = torch.from_numpy(
            train_disease_multilabels
        ).float()
        tensor_class_multilabel_valid = torch.from_numpy(
            valid_disease_multilabels
        ).float()

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

        if CLS_MULTILABEL:
            tensor_class_multilabel_train = tensor_class_multilabel_train[
                train_sort_ids
            ]
            tensor_class_multilabel_valid = tensor_class_multilabel_valid[
                valid_sort_ids
            ]

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

    if CLS_MULTILABEL:
        train_data_pt["class_multilabel"] = tensor_class_multilabel_train
        valid_data_pt["class_multilabel"] = tensor_class_multilabel_valid

    return train_data_pt, valid_data_pt


class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


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


# endregion

# region 1. Specify hyper-parameter setup for integration task

hyperparameter_defaults = dict(
    seed=0,
    dataset_name="test_1",
    do_train=True,
    # load_model="../save/scGPT_human",
    load_model="/aloy/home/ddalton/projects/scGPT_playground/save/scGPT_human",
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
with open(
    "/aloy/home/ddalton/projects/scGPT_playground/scripts/config/wandb.json", "r"
) as f:
    api_config = json.load(f)

# Use the Wandb API key from the config file
if "wandb_api_key" in api_config:
    os.environ["WANDB_API_KEY"] = api_config["wandb_api_key"]

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
output_style = "binned"  # "normed_raw", "log1p", or "binned"manual_parameters.ADV

# updated scGPT settings
output_attentions = manual_parameters.get("output_attentions")

print("original ecs threshold", config.ecs_thres)
print("original dab weight", config.dab_weight)

# settings for training
MLM = manual_parameters.get(
    "MLM"
)  # whether to use masked language modeling, currently it is always on.
CLS = manual_parameters.get("CLS")  # celltype classification objective
CLS_MULTILABEL = manual_parameters.get(
    "CLS_multilabel"
)  # celltype classification objective, multilabel
ADV = manual_parameters.get("ADV")  # Adversarial training for batch correction
CCE = manual_parameters.get("CCE")  # Contrastive cell embedding objective
MVC = config.MVC  # Masked value prediction for cell embedding
# ECS = config.ecs_thres > 0  # Elastic cell similarity objective
ECS = manual_parameters.get("ecs_thres") > 0  # Elastic cell similarity objective
DAB = manual_parameters.get(
    "DAB"
)  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
INPUT_BATCH_LABELS = manual_parameters.get(
    "INPUT_BATCH_LABELS"
)  # TODO: have these help MLM and MVC, while not to classifier
input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"
adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
adv_D_delay_epochs = 0
mvc_decoder_style = "inner product"
ecs_threshold = manual_parameters.get("ecs_thres")
# dab_weight = config.dab_weight
dab_weight = manual_parameters.get("dab_weight")

explicit_zero_prob = MLM and include_zero_gene  # whether explicit bernoulli for zeros
do_sample_in_train = False and explicit_zero_prob  # sample the bernoulli in training

per_seq_batch_sample = False

# settings for optimizer
lr = config.lr  # TODO: test learning rate ratio between two tasks
lr_ADV = 1e-3  # learning rate for discriminator, used when ADV is True

batch_size = manual_parameters.get("batch_size")
eval_batch_size = batch_size
schedule_interval = 1

# settings for the model
# fast_transformer = config.fast_transformer
use_fast_transformer = manual_parameters.get(
    "use_fast_transformer"
)  # if using output_attentions not use fast_transformer
# and vice versa

fast_transformer_backend = "flash"  # "linear" or "flash"
embsize = config.layer_size  # embedding dimension
d_hid = config.layer_size  # dimension of the feedforward network in TransformerEncoder
nlayers = config.nlayers  # number of TransformerEncoderLayer in TransformerEncoder
nhead = config.nhead  # number of heads in nn.MultiheadAttention
dropout = config.dropout  # dropout probability
print(f"Trained dropout is {dropout}")

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

# endregion

# region 2. Load and pre-process data

# We follow the standard scGPT data pre-processing pipelines for the cell-type annotation task. Note that since now we have two datasets at hand (i.e., reference and query data), the same pre-prpocessing steps need to be applied to both of them.
# load data
adata = sc.read(manual_parameters.get("data_path"))
print("adata loaded")
print(adata.X)


#! REMOVE - THIS IS A QUICK AND UGLY FIX
only_control = True
if only_control:
    adata = adata[adata.obs["doid_id"] != "Control"].copy()

# define celltype as disease
if manual_parameters.get("ontology") == "mesh":
    adata.obs["celltype"] = adata.obs["mesh_disease"].astype("category")
elif manual_parameters.get("ontology") == "do":
    print(adata.obs.columns)
    # adata.obs["celltype"] = adata.obs["do_term"].astype("category")
    adata.obs["celltype"] = adata.obs["do_id"].astype("category")

# generate celltype label
celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
celltypes = adata.obs["celltype"].unique()
num_types = len(np.unique(celltype_id_labels))
id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
adata.obs["celltype_id"] = celltype_id_labels

# config parameters
data_is_raw = True
filter_gene_by_counts = False


# gene filtering
if manual_parameters.get("gene_filtering") == "top_presence":
    mask_genes = tr_h.get_top_k_most_present_genes(
        adata, k=manual_parameters.get("max_seq_len")
    )
elif manual_parameters.get("gene_filtering") == "top_variance":
    mask_genes = tr_h.get_top_k_highest_variance_genes(
        adata, k=manual_parameters.get("max_seq_len")
    )
elif manual_parameters.get("gene_filtering") == "random":
    mask_genes = tr_h.get_k_random_genes(adata, k=manual_parameters.get("max_seq_len"))
elif manual_parameters.get("gene_filtering") == "top_variance_high_presence":
    mask_genes = tr_h.get_top_k_highest_variance_genes_with_presence(
        adata,
        k=manual_parameters.get("max_seq_len"),
        presence_pct=manual_parameters.get("gene_presence_pct"),
    )


logging.info(f"Combined mask {np.sum(mask_genes)} genes left")

# mask the genes
adata = adata[:, mask_genes]


# mask samples
# non_nan_percentage = np.sum(~np.isnan(adata.X), axis=1) / adata.X.shape[1]
non_zero_non_nan_mask = ~np.isnan(adata.X) & ~(adata.X == 0)

non_zero_non_nan_mask_pct = np.sum(non_zero_non_nan_mask, axis=1) / adata.X.shape[1]

# mask samples that have less than 30% non-NaN values
mask_samples = non_zero_non_nan_mask_pct >= manual_parameters.get("sample_presence_pct")
logging.info(
    f"Filtering out {np.sum(~mask_samples)} / {len(mask_samples)} samples with less than 30% non-NaN values"
)

# apply the mask to the AnnData object
adata = adata[mask_samples, :]


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


#! ADDED
if CLS_MULTILABEL:
    """If CLS multilabel genearte new celltype_id labels - this will be used
    to generate multilabel classification loss!
    """

    #! WE ARE HERE TESTING WITH LOG2CPM COUNTS
    #! No need to normalize

    # preprocessor = Preprocessor(
    #     use_key="X",                 # or the layer name where your log2CPM lives
    #     filter_gene_by_counts=False, # don't use count-based filters on logged data
    #     filter_cell_by_counts=False, # same reason
    #     normalize_total=False,       # already library-size normalized (CPM)
    #     result_normed_key=None,      # irrelevant since normalize_total=False
    #     log1p=False,                 # already log-transformed (base-2)
    #     result_log1p_key=None,       # irrelevant since log1p=False
    #     subset_hvg=0,                # or an int like 2000 if you want HVG selection
    #     hvg_use_key="X",             # use your log2CPM layer for HVG calc
    #     hvg_flavor=n_bins,    # better when data is already normalized/logged
    #     binning="X_binned"                 # leave None unless your model *requires* bins
    # )

    try:
        import obonet
    except ImportError:
        import subprocess

        print("obonet not found. Installing locally...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--user", "obonet"]
        )
        import obonet

    # imports
    sys.path.append("../../")
    from src.utils import utils as u

    # get Disease Ontology graph
    do_g = u.load_do_graph()

    # get sanchez IC
    doid_2_ic = u.get_sanchez_ic(do_g)

    #! IMPORTANT - FIX CONTROL CLASS

    # get nodes
    _class_nodes = u.get_lvl1_nodes(do_g)
    # _class_nodes = u.get_n_lowest_ic_nodes(doid_2_ic, 50)
    # benchmark class nodes - use the sames as in the single classifier benchmark
    # _class_nodes = adata.obs["do_id"].unique().tolist()

    # Generate multilabel vectors for level 1 nodes
    Y_multilabel, _class_nodes = u.generate_multilabel_vectors(
        adata, do_g, _class_nodes
    )
    print(
        f"Generated multilabel vectors for level 1 nodes with shape {Y_multilabel.shape}"
    )

    # Clean multilabel vectors by removing nodes with no samples
    Y_multilabel, _class_nodes = u.clean_multilabel_vectors(Y_multilabel, _class_nodes)
    print(
        f"Cleaned multilabel vectors for top 50 nodes with shape {Y_multilabel.shape}"
    )

    # Check the multilabel vector for level 1 nodes
    u.check_multilabel_vector(Y_multilabel, _class_nodes, do_g)

    # get doids and class names
    Y_multilabel_doid, Y_multilabel_name = u.get_multilabel_data(
        Y_multilabel, _class_nodes, do_g
    )

    # add multilabel vectors to adata
    #! Note: we can add and extract arrays from the data frame - but not save them !
    adata.obs = u.add_multilabel_to_adata(
        adata, Y_multilabel, Y_multilabel_doid, Y_multilabel_name
    )

    print(
        f"Nº of times each class appears in the multilabel vector:\n{np.sum(Y_multilabel, axis=0)}"
    )

    # compute pos_weight for BCEWithLogitsLoss
    pos_weight = None
    if True:
        pos_weight = tr_h.get_pos_weight(Y_multilabel)
        print(
            f"Generated poitional weights for BCEWithLogitsLoss with shape {pos_weight.shape}"
        )
        print(f"Positional weights:\n{pos_weight}")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pos_weight = pos_weight.to(device) if pos_weight is not None else None

    # modify the nº of classes
    num_types = len(_class_nodes)
    print(f"Number of classes: {num_types}")


#! WHAT IS THIS
print("config.load_model", config.load_model)
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

if manual_parameters.get("split_type") == "stratified":
    # split
    df_obs = adata.obs
    new_obs = tr_h.split_stratified(
        df=df_obs,
        y_label="celltype",
        group_label="dataset_id",
        split_size=manual_parameters.get("n_splits"),
        seed=manual_parameters.get("seed"),
    )
    tr_h.report_split(new_obs)

    # update observations with new column split
    adata.obs = new_obs

elif manual_parameters.get("split_type") == "non_stratified":
    df_obs = adata.obs
    new_obs = tr_h.get_test_split(
        obs=df_obs, n_splits=manual_parameters.get("n_splits")
    )
    adata.obs = new_obs

elif manual_parameters.get("split_type") == "mixed":
    df_obs = adata.obs
    new_obs = tr_h.get_test_split_common(
        obs=df_obs, n_splits=manual_parameters.get("n_splits")
    )
    adata.obs = new_obs

# store original data
adata_orig = adata.copy()

# dictionary to store the results
data_to_save = {
    "split": list(),
    "predictions_test": list(),
    "labels_test": list(),
    "results_test": list(),
    "all_outputs_test": list(),
    "predictions_valid": list(),
    "labels_valid": list(),
    "results_valid": list(),
    "all_outputs_valid": list(),
    "predictions_train": list(),
    "labels_train": list(),
    "results_train": list(),
    "all_outputs_train": list(),
    "id2type": list(),
    "adata_orig": adata_orig,
    "train_indices": list(),
    "valid_indices": list(),
    "d_metrics": dict(),
}

# Generate the output directory
output_dir = u.generate_output_folder_dir()
d_metrics = dict()

for split in range(1, manual_parameters.get("n_tested_splits") + 1):

    d_metrics[f"split_{split}"] = {
        "train_loss": list(),
        "valid_loss": list(),
        "train_acc": list(),
        "valid_acc": list(),
        "train_err": list(),
        "valid_err": list(),
        "test_err": list(),
    }

    torch.cuda.empty_cache()

    #! ADDED - BATCH CORRECTION OPTION
    #! ASSUMING GAUSSIAN DISTRIBUTION IN USE OF COMBAT! REVISE!
    if manual_parameters.get("do_combat"):
        # preprocess & batch correct
        preprocessor(adata, batch_key=None)

        # If we have samples from 1 single batch - combat will return NaN values for all
        # we tackle this by removing samples with a single batch id
        # remove samples from batch_id with only one sample
        print(f"Shape of adata before preprocessing: {adata.shape}")
        batch_counts = dict(adata.obs["batch_id"].value_counts())
        _low_count_batches = [k for k, v in batch_counts.items() if v < 2]
        mask = adata.obs["batch_id"].isin(_low_count_batches)
        adata = adata[~mask].copy()
        print(f"Removed {len(_low_count_batches)} batches with less than 2 samples")
        print(f"Shape of _adata after removing low count batches: {adata.shape}")

        adata = tr_h.perform_combat_correction(adata)
        print("adata after preprocessing and batch correction")
        print(adata.X)

        batch_ids = adata.obs["batch_id"].tolist()
        num_batch_types = len(set(batch_ids))
        batch_ids = np.array(batch_ids)

        """Re-map batch ids so it matches the max value of batches
        """
        _remap_dict = {k: i for i, k in enumerate(sorted(set(batch_ids)))}
        batch_ids = np.array([_remap_dict[b] for b in batch_ids], dtype=int)
        adata.obs["batch_id"] = batch_ids  # update the batch ids in adata.obs

        # seperate the test and train data
        adata_test = adata[adata.obs[f"test_split_{split}"] == 1]
        adata = adata[adata.obs[f"test_split_{split}"] == 0]

        print("adata_test after preprocessing and batch correction")
        print(adata_test.X)

    else:

        batch_ids = adata.obs["batch_id"].tolist()
        num_batch_types = len(set(batch_ids))
        batch_ids = np.array(batch_ids)

        """Re-map batch ids so it matches the max value of batches
        """
        _remap_dict = {k: i for i, k in enumerate(sorted(set(batch_ids)))}
        batch_ids = np.array([_remap_dict[b] for b in batch_ids], dtype=int)
        adata.obs["batch_id"] = batch_ids  # update the batch ids in adata.obs

        # seperate data
        adata_test = adata_orig[adata_orig.obs[f"test_split_{split}"] == 1]
        adata = adata_orig[adata_orig.obs[f"test_split_{split}"] == 0]

        # added
        adata_test_raw = adata_test.copy()

        # batch correct - same as in original tutorial
        preprocessor(adata, batch_key=None)
        preprocessor(adata_test, batch_key=None)

    #! ASSESS MAX VALUES AFTER PP

    input_layer_key = (
        {  # the values of this map coorespond to the keys in preprocessing
            "normed_raw": "X_normed",
            "log1p": "X_normed",
            "binned": "X_binned",
        }[input_style]
    )
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )
    genes = adata.var["gene_name"].tolist()

    celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
    celltypes_labels = np.array(celltypes_labels)

    # Create indices for the entire dataset
    all_indices = np.arange(len(all_counts))

    # Split to get indices only
    train_idx, valid_idx = train_test_split(
        np.arange(len(all_counts)),
        test_size=0.1,
        shuffle=True,
        stratify=celltypes_labels,
    )

    # Use indices to split the data manually
    train_data = all_counts[train_idx]
    valid_data = all_counts[valid_idx]

    train_celltype_labels = celltypes_labels[train_idx]
    valid_celltype_labels = celltypes_labels[valid_idx]

    train_batch_labels = batch_ids[train_idx]
    valid_batch_labels = batch_ids[valid_idx]

    train_indices = all_indices[train_idx]
    valid_indices = all_indices[valid_idx]

    if CLS_MULTILABEL:
        d_y_multilabel = u.get_multilabel_dict_from_adata(adata)
        _Y_multilabel = np.array(d_y_multilabel["Y_multilabel"])
        # Get the multilabel vectors for the training and validation sets
        train_disease_multilabels = _Y_multilabel[train_idx]
        valid_disease_multilabels = _Y_multilabel[valid_idx]

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

    # region 3. Load the pre-trained scGPT model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ntokens = len(vocab)  # size of vocabulary
    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        # output_attentions=output_attentions,
        nlayers_cls=3,
        n_cls=num_types if (CLS or CLS_MULTILABEL) else 1,
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
        use_fast_transformer=use_fast_transformer,
        fast_transformer_backend=fast_transformer_backend,
        pre_norm=config.pre_norm,
    )

    # print("\n[INFO] Model parameters and config at initialization:\n")
    # for name, param in model.named_parameters():
    #     print(f"{name:60} shape: {tuple(param.shape)} requires_grad: {param.requires_grad}")
    # sys.exit(0)

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
    if CLS_MULTILABEL:
        criterion_cls_multilabel = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight, reduction="mean"
        )
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

    # endregion

    # region 4. Finetune scGPT with task-specific objectives

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
    tr_h.define_wandb_metrcis()

    # quality check
    """Check if there are no diseases or labels with empty labels!"""

    d_y_multilabel = u.get_multilabel_dict_from_adata(adata_test)
    _Y_test = np.array(d_y_multilabel["Y_multilabel"])

    train_adata = adata[train_indices, :]  # Subset adata for training samples
    valid_adata = adata[valid_indices, :]  # Subset adata for validation samples

    d_y_multilabel = u.get_multilabel_dict_from_adata(valid_adata)
    _Y_valid = np.array(d_y_multilabel["Y_multilabel"])
    
    d_y_multilabel = u.get_multilabel_dict_from_adata(train_adata)
    _Y_train = np.array(d_y_multilabel["Y_multilabel"])


    print("_Y_test", _Y_test.shape, min(np.sum(_Y_test, axis=0)), min(np.sum(_Y_test, axis=1)) )
    print("_Y_valid", _Y_valid.shape, min(np.sum(_Y_valid, axis=0)), min(np.sum(_Y_valid, axis=1)) )
    print("_Y_train", _Y_train.shape, min(np.sum(_Y_train, axis=0)), min(np.sum(_Y_train, axis=1)) )


    # sys.exit(0)


    epochs = manual_parameters.get("epochs")
    patience = 5
    wait = 0
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

        print(">>> Finished remapping batch labels")
        print(f"Total unique remapped batch labels: {adata.obs['batch_id'].nunique()}")
        print(
            f"Range of remapped batch labels: {adata.obs['batch_id'].min()} to {adata.obs['batch_id'].max()}"
        )
        print(
            ">>> Sample batch labels after remapping:",
            adata.obs["batch_id"].value_counts().head(),
        )

        print(f"CLS is {CLS}")

        if config.do_train:
            # train
            train_loss, train_err = train(
                model,
                loader=train_loader,
            )

        # validation
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

        d_metrics[f"split_{split}"]["train_loss"].append(train_loss)
        d_metrics[f"split_{split}"]["valid_loss"].append(val_loss)
        d_metrics[f"split_{split}"]["train_err"].append(train_err)
        d_metrics[f"split_{split}"]["valid_err"].append(val_err)
        d_metrics[f"split_{split}"]["train_acc"].append(1 - train_err)
        d_metrics[f"split_{split}"]["valid_acc"].append(1 - val_err)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            logger.info(f"Best model with score {best_val_loss:5.4f}")
        else:
            logger.info("Validation loss did not improve.")
            logger.info(
                f"Current best model score: {best_val_loss:5.4f} at epoch {best_model_epoch}"
            )
            wait += 1
            if wait >= patience:
                print("early stop")
                break

        scheduler.step()
        if DAB_separate_optim:
            scheduler_dab.step()
        if ADV:
            scheduler_D.step()
            scheduler_E.step()

        #! FOR DEBUGGING
        # break

    # ## Step 5: Inference with fine-tuned scGPT model
    # In the cell-type annotation task, the fine-tuned scGPT predicts cell-type labels for query set as inference. The model performance is evaluated on standard classificaton metrics. Here we visualize the predicted labels over the scGPT cell embeddings, and present the confusion matrix for detailed classification performance on the cell-group level.

    # test split
    predictions, labels, results = test(best_model, adata_test)

    ## Evaluate the model on the train set

    predictions_train, labels_train, results_train = test(best_model, adata)

    # ### Evaluate results on train

    logging.info("region 4")
    tr_h.log_cpu_memory_usage()
    tr_h.log_gpu_memory_usage()

    # endregion

    # region 5. Inference with fine-tuned scGPT model
    adata_train_raw = adata.copy()

    # get predictions
    # test inference
    print(f"####\tTest inference for split {split}\t####")
    (
        predictions_test,
        labels_test,
        results_test,
        all_outputs_test,
    ) = test_2(best_model, adata_test)

    logging.info(f"Results Test: {results_test}")
    # Assume train_indices and valid_indices have been loaded for the current split
    train_adata = adata[train_indices, :]  # Subset adata for training samples
    valid_adata = adata[valid_indices, :]  # Subset adata for validation samples

    print("train_adata shape:", train_adata.shape)
    print(train_adata.X)

    # Perform inference on train and validation sets
    # Train inference
    print(f"####\tTraining inference for split {split}\t####")
    (
        predictions_train,
        labels_train,
        results_train,
        all_outputs_train,
    ) = test_2(best_model, train_adata)

    # Validation inference
    print(f"####\tValidation inference for split {split}\t####")
    (
        predictions_valid,
        labels_valid,
        results_valid,
        all_outputs_valid,
    ) = test_2(best_model, valid_adata)

    logging.info("region 5")
    tr_h.log_cpu_memory_usage()
    tr_h.log_gpu_memory_usage()

    # endregion

    # region 6. Save output
    # Prepare a dictionary of all variables to be saved
    data_to_save["split"].append(split)
    data_to_save["predictions_test"].append(predictions_test)
    data_to_save["labels_test"].append(labels_test)
    data_to_save["results_test"].append(results_test)
    data_to_save["all_outputs_test"].append(all_outputs_test)
    data_to_save["predictions_valid"].append(predictions_valid)
    data_to_save["labels_valid"].append(labels_valid)
    data_to_save["results_valid"].append(results_valid)
    data_to_save["all_outputs_valid"].append(all_outputs_valid)
    data_to_save["predictions_train"].append(predictions_train)
    data_to_save["labels_train"].append(labels_train)
    data_to_save["results_train"].append(results_train)
    data_to_save["all_outputs_train"].append(all_outputs_train)
    data_to_save["id2type"].append(id2type)
    data_to_save["metrics_epochs"] = d_metrics

    # Save the train/valid indices for this split
    data_to_save[f"train_indices"].append(train_indices.tolist())
    data_to_save[f"valid_indices"].append(valid_indices.tolist())

    # Store processed adata objects
    data_to_save[f"adata_train_{split}"] = train_adata
    data_to_save[f"adata_valid_{split}"] = valid_adata
    data_to_save[f"adata_test_{split}"] = adata_test

    # save best model
    torch.save(best_model, os.path.join(output_dir, f"model_{split}.pt"))

    break

# Save each item in the dictionary to a pickle file
for filename, data in data_to_save.items():
    if filename.startswith("adata"):
        if CLS_MULTILABEL:
            # we have to remove multilabel daata so it does not cause issues
            _d_multilabel = u.get_multilabel_dict_from_adata(data)
            # save to pickle
            _file_path_pkl = os.path.join(output_dir, f"{filename}.multilabels.pkl")
            with open(_file_path_pkl, "wb") as f:
                pickle.dump(_d_multilabel, f)

            # remove the columns corresponding to multilabels
            data.obs.drop(
                columns=[
                    "class_multilabel_doid",
                    "class_multilabel_name",
                    "class_multilabel",
                ],
                inplace=True,
            )

            # save
            file_path = os.path.join(output_dir, f"{filename}.h5ad")
            data.write(file_path)

        else:
            file_path = os.path.join(output_dir, f"{filename}.h5ad")
            data.write(file_path)
    else:
        file_path = os.path.join(output_dir, f"{filename}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(data, f)


logging.info("region 4")
tr_h.log_cpu_memory_usage()
tr_h.log_gpu_memory_usage()

# save manual parameters
# Write parameters to a JSON file
with open(os.path.join(output_dir, "parameters.json"), "w") as json_file:
    json.dump(manual_parameters, json_file, indent=4)

# save vocab
vocab_file = os.path.join(output_dir, "vocab.json")
vocab.save_json(vocab_file)


# save model config
model_config_file = os.path.join(output_dir, "args.json")

model_configs = {
    "embsize": embsize,
    "nheads": nhead,
    "d_hid": d_hid,
    "nlayers": nlayers,
    # "n_layers_cls": n_layers_cls
}
with open(model_config_file, "w") as f:
    json.dump(model_configs, f, indent=4)
