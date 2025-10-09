"""Get Similarities

Structure:
    1. Imports, Variables, Functions
    2. Load Data
    3. Compute Similarities
    4. Save Results
"""
# 1. Imports, Variables, Functions
# imports 
from typing import *
from scipy.spatial.distance import cdist    
import itertools
import obonet
import networkx as nx
import sys
sys.path.append(os.path.join("..", ".."))
from src.utils import utils as ut
from src.utils import viz as vz
import os, pickle, numpy as np, scanpy as sc
import argparse

parser = argparse.ArgumentParser(description="Script for scGPT project")

# variables
parser.add_argument("--embedding_type", type=str, required=True, help="Path to the data file" )
# parser.add_argument("--pad_nan_vals", type=eval, default=False, help="PAD NaN values in input" )
args = parser.parse_args()

# functions

# 2. Load Data

# load scGPT embeddings
print(f"Using {args.embedding_type} embeddings")

if args.embedding_type == "ft":
    all_outputs_test = pickle.load(open(f"{args.run_dir}all_outputs_test.pkl", "rb"))
    all_outputs_valid = pickle.load(open(f"{args.run_dir}all_outputs_valid.pkl", "rb"))
    all_outputs_train = pickle.load(open(f"{args.run_dir}all_outputs_train.pkl", "rb"))

    embeddings_test = vz.merge_embeddings(all_outputs_test[0])
    embeddings_valid = vz.merge_embeddings(all_outputs_valid[0])
    embeddings_train = vz.merge_embeddings(all_outputs_train[0])

elif args.embedding_type == "pt":
    adata_test = sc.read(os.path.join(args.run_dir.replace("/aloy/","/scratch/"), f"adata_test_1.h5ad"), backed="r")
    adata_valid = sc.read(os.path.join(args.run_dir.replace("/aloy/","/scratch/"), f"adata_valid_1.h5ad"), backed="r")
    adata_train = sc.read(os.path.join(args.run_dir.replace("/aloy/","/scratch/"), f"adata_train_1.h5ad"), backed="r")

    embeddings_test = adata_test.obsm["pt_scGPT"]
    embeddings_valid = adata_valid.obsm["pt_scGPT"]
    embeddings_train = adata_train.obsm["pt_scGPT"]
elif args.embedding_type == "raw":
    
    adata_test =  sc.read(
    os.path.join(args.run_dir, f"adata_test_1.h5ad"), backed="r"
    )
    adata_test.obs.reset_index(drop=True, inplace=True)  
    adata_valid =  sc.read(
    os.path.join(args.run_dir, f"adata_valid_1.h5ad"), backed="r"
    )
    adata_valid.obs.reset_index(drop=True, inplace=True)  
    adata_train =  sc.read(
    os.path.join(args.run_dir, f"adata_train_1.h5ad"), backed="r"
    )
    adata_train.obs.reset_index(drop=True, inplace=True)  
    
    embeddings_test = adata_test.X
    embeddings_valid = adata_valid.X
    embeddings_train = adata_train.X


# compute cosine similarity between all embeddings
c_matrix_train = 1 - cdist(embeddings_train, embeddings_train, "cosine")
c_matrix_valid = 1 - cdist(embeddings_valid, embeddings_valid, "cosine")
c_matrix_test = 1 - cdist(embeddings_test, embeddings_test, "cosine")

# compute cosine similarity between sets
c_matrix_test_vs_train = 1 - cdist(embeddings_test, embeddings_train, "cosine")
c_matrix_valid_vs_train = 1 - cdist(embeddings_valid, embeddings_train, "cosine")

# convert all to float16 to save space
c_matrix_train = c_matrix_train.astype(np.float16)
c_matrix_valid = c_matrix_valid.astype(np.float16)
c_matrix_test = c_matrix_test.astype(np.float16)
c_matrix_test_vs_train = c_matrix_test_vs_train.astype(np.float16)
c_matrix_valid_vs_train = c_matrix_valid_vs_train.astype(np.float16)


# 4. Save Results
pickle.dump(c_matrix_train, open(os.path.join(args.run_dir.replace("/aloy/", "/scratch/"), f"c_matrix_train_{args.embedding_type}.pkl"), "wb"))
pickle.dump(c_matrix_valid, open(os.path.join(args.run_dir.replace("/aloy/", "/scratch/"), f"c_matrix_valid_{args.embedding_type}.pkl"), "wb"))
pickle.dump(c_matrix_test, open(os.path.join(args.run_dir.replace("/aloy/", "/scratch/"), f"c_matrix_test_{args.embedding_type}.pkl"), "wb"))
pickle.dump(c_matrix_test_vs_train, open(os.path.join(args.run_dir.replace("/aloy/", "/scratch/"), f"c_matrix_test_vs_train_{args.embedding_type}.pkl"), "wb"))
pickle.dump(c_matrix_valid_vs_train, open(os.path.join(args.run_dir.replace("/aloy/", "/scratch/"), f"c_matrix_valid_vs_train_{args.embedding_type}.pkl"), "wb"))
