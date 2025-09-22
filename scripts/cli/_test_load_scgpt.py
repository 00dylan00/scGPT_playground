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

# ===== Third-party =====
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

import torch, sys
# (optional) make sure your scGPT source is importable if you need to instantiate the model
sys.path.insert(0, "/aloy/home/ddalton/git_clones/scGPT")

model_path = "/aloy/home/ddalton/projects/scGPT_playground/outputs/run-25-09-17-01/model_1.pt"

model = torch.load(model_path, map_location="cpu")
print(model)
print("Model loaded successfully")

torch.save(model.state_dict(), "/aloy/home/ddalton/projects/scGPT_playground/outputs/run-25-09-17-01/model_1.state_dict.pt")
print("State dict saved successfully")
