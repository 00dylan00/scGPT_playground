# imports
import os
import obonet
import networkx as nx
import math
from datetime import datetime
import numpy as np
import scanpy as sc
from typing import *
import pandas as pd
import logging, pickle
logging.basicConfig(level=logging.INFO)

def load_do_graph():
    """Load the Disease Ontology graph from the OBO file."""
    do_obl_data_path = os.path.join(
        "/aloy/home/ddalton/projects/disease_signatures",
        "data",
        "DiseaseOntology",
        "doid.obo",
    )
    do_graph = obonet.read_obo(do_obl_data_path)
    do_graph = nx.DiGraph(do_graph)
    do_graph = do_graph.reverse()   # reverse so leafs are at the bottom  
    return do_graph

def load_dsa_info()->pd.DataFrame:
    """
    Load the DSAID information DataFrame.
    """
    return pd.read_csv(os.path.join(
    "/aloy/home/ddalton/projects/disease_signatures",
    "data",
    "DiSignAtlas",
    "Disease_information_Datasets.csv",
    ))

def get_umls_2_doid_mapping(do_G: nx.DiGraph)-> dict:
    """Build a mapping from UMLS CUI to DOID"""    

    # Build UMLS CUI → DOID mapping
    umls_2_doid = {}

    for node_id, data in do_G.nodes(data=True):
        xrefs = data.get('xref', [])
        for ref in xrefs:

            if ref.startswith('UMLS_CUI:'):
                umls_cui = ref.split(':', 1)[1]
                umls_2_doid[umls_cui] = node_id

    return umls_2_doid

def get_sanchez_ic(do_G: nx.DiGraph) -> dict:
    """Compute Information Content (IC) using Sánchez formula."""

    # Identify leaves (nodes with no children)
    do_leaves = [n for n in do_G.nodes() if do_G.out_degree(n) == 0]
    N_do_leaves = len(do_leaves)
    print(f"Number of DO leaves: {N_do_leaves}")


    # Compute IC for each node using Sánchez formula
    do_sanchez_ic = {}

    for node in do_G.nodes:
        # Descendant leaves
        descendants = nx.descendants(do_G, node)
        leaf_desc = [n for n in descendants if n in do_leaves]
        if node in do_leaves:
            leaf_desc.append(node)
        num_leaf_desc = len(leaf_desc)

        # Subsumers (ancestors + self)
        subsumers = nx.ancestors(do_G, node)
        subsumers.add(node)
        num_subsumers = len(subsumers)

        ic = -math.log((num_leaf_desc + 1) / ((num_subsumers + 1) * (N_do_leaves + 1)))
        do_sanchez_ic[node] = ic

    return do_sanchez_ic

def add_multilabel_to_adata(adata: sc.AnnData, Y_multilabel_50: np.ndarray, Y_multilabel_50_doid: list, Y_multilabel_50_name: list) -> sc.AnnData:
    """Add multilabel vectors to AnnData object."""
    adata_obs = adata.obs.copy()
    adata_obs["class_multilabel_doid"] = list(Y_multilabel_50_doid)
    adata_obs["class_multilabel_name"] = list(Y_multilabel_50_name)
    adata_obs["class_multilabel"] = list(Y_multilabel_50)
    return adata_obs

def get_multilabel_dict_from_adata(adata: sc.AnnData) -> dict:
    """Get multilabel data from AnnData object."""
    Y_multilabel = adata.obs.get("class_multilabel").to_list()
    Y_multilabel_doid = adata.obs.get("class_multilabel_doid").to_list()
    Y_multilabel_name = adata.obs.get("class_multilabel_name").to_list()
    return {
        "Y_multilabel": Y_multilabel,
        "Y_multilabel_doid": Y_multilabel_doid,
        "Y_multilabel_name": Y_multilabel_name
    }

def get_multilabel_data(Y_multilabel: np.ndarray, all_nodes: list, do_g: nx.Graph) -> tuple[list, list]:
    """Get from multilabel data from the Y_multilabel array."""
    Y_multilabel_doid = [all_nodes[Y_multilabel[i].astype(bool)] for i in range(Y_multilabel.shape[0])]
    Y_multilabel_name = list()
    for nodes in Y_multilabel_doid:
        names = [do_g.nodes[n].get("name", "Unknown") for n in nodes]
        Y_multilabel_name.append(names)

    return Y_multilabel_doid, Y_multilabel_name

def get_lvl1_nodes(do_g: nx.Graph) -> list:
    # get root node
    root_nodes = [n for n in do_g.nodes if do_g.in_degree(n) == 0]
    assert len(root_nodes) == 1, "Multiple root nodes found!"
    root = root_nodes[0]
    print(f"Root node: {root} - {do_g.nodes[root].get('name', 'Unknown')}")

    # get level 1 
    lvl1_nodes = list(do_g.successors(root))  # First level children
    print(f"Nº Level 1 nodes: {len(lvl1_nodes)}")

    return lvl1_nodes

def get_n_lowest_ic_nodes(doid_2_ic: dict, n: int = 50) -> list:
    """Get the n lowest IC nodes from the doid_2_ic dictionary.
    Remove root"""
    # get lowest 50 nodes by Sanchez IC
    bot_n_nodes = sorted(doid_2_ic, key=doid_2_ic.get, reverse=False)[:n+1]
    bot_n_nodes.remove("DOID:4")  # remove root node
    return bot_n_nodes


def check_multilabel_vector(Y_multilabel, nodes, do_g)-> None:
    _sum_nodes = np.sum(Y_multilabel, axis=0)
    for i, _node in enumerate(nodes):
        print(f"{Y_multilabel[:, i].sum()} samples\tNode {_node} - {do_g.nodes[_node].get('name', 'Unknown')}")

    _sum_samples = np.sum(Y_multilabel, axis=1)
    print(f"Nº of samples with only one label: {np.sum(_sum_samples == 1)}")
    print(f"Nº of samples with +1 labels: {np.sum(_sum_samples > 1)}")
    print(f"Max nº of labels per sample: {np.max(_sum_samples)}")

def generate_multilabel_vectors(adata:sc.AnnData, do_g:nx.graph, nodes:list)->tuple[np.ndarray, np.ndarray]:
    """Generate multilabel vectors for the given AnnData object based on the Disease Ontology graph."""
    from sklearn.preprocessing import MultiLabelBinarizer

    # define class nodes
    nodes = sorted(nodes)

    # For each sample, collect all ancestors
    # then assess if they are in the class nodes
    sample_class_nodes = []
    for doid in adata.obs["do_id"]:

        if doid.lower() == "Control":
            # If the sample is a control, assign a single class node as Control
            #! IMPORTANT - WE MIGHT WANT TO CHANGE THIS TO BE A STRATIFIED CONTRO + DATASET LABEL
            sample_class_nodes.append(["Control"])
            continue
        
        ancestors = nx.ancestors(do_g, doid)
        ancestors.add(doid) # include itself just in case

        # Keep only those in class nodes
        _present_class_nodes = list(ancestors.intersection(nodes))
        sample_class_nodes.append(_present_class_nodes)

    # Fit MultiLabelBinarizer on the fixed level 1 vocabulary
    mlb = MultiLabelBinarizer(classes=nodes)
    Y_multilabel = mlb.fit_transform(sample_class_nodes)

    return Y_multilabel, np.array(nodes)


def clean_multilabel_vectors(Y_multilabel:np.ndarray, nodes:np.ndarray,)->tuple[np.ndarray,np.ndarray]:
    """Clean multilabel vectors by removing nodes with no samples."""
    # Identify nodes with no samples
    node_sums = np.sum(Y_multilabel, axis=0)
    mask_keep = node_sums > 0

    cleaned_Y = Y_multilabel[:, mask_keep]
    cleaned_nodes = np.array(nodes)[mask_keep]

    return cleaned_Y, cleaned_nodes

def get_processed_ids()-> list:
    """Get processed ids
    Returns:
        list: list of processed ids
    """
    data_path = os.path.join(
        "/aloy/home/ddalton/projects/disease_signatures",
        "data",
        "DiSignAtlas",
        "dsa_diff_download.processed",
    )
    return [f.split("_")[0] for f in os.listdir(data_path)]

def get_human_entrez_protein_coding_ids()-> list:
    """Get Human Entrez IDs
    Returns:
        list: list of human entrez ids
    """
    data_path = os.path.join(
        "/aloy/home/ddalton/projects/disease_signatures",
        "data",
        "ncbi_gene_info",
        "gene_info",
    )
    df = pd.read_csv(data_path, sep="\t", usecols=["#tax_id", "GeneID", "type_of_gene"])
    df_human = df[(df["#tax_id"] == 9606) & (df["type_of_gene"] == "protein-coding")]
    logging.info(f"Nº of Human protein coding genes: {len(df_human)}")
    return df_human["GeneID"].to_list()

def load_gene_signatures()->dict:
    import os, pickle, logging
    # load signatures
    path_pkl = os.path.join(
        "/aloy/home/ddalton/projects/disease_signatures/",
        "data",
        "DiSignAtlas",
        "signatures.pkl",
    )
    logging.info(f"Loading signatures from file {path_pkl}")
    signatures = pickle.load(open(path_pkl, "rb"))
    clean_signatures = {}
    for s in signatures:
        clean_signatures[s[0]] = {"gene_ids":s[2],
                                  "log2FC":s[5],
                                  "pval":s[3],
                                  "adj_pval":s[4],
                                  } 
    return clean_signatures

def get_dsaid_2_doid_mapping()-> dict:
    """
    Get a mapping from DSAID to DOID.
    """
    
    # map DSAIDs to DOIDs using UMLS codes
    external_links_data_path = os.path.join(
        "/aloy/home/ddalton/projects/disease_signatures",
        "data",
        "DiSignAtlas",
        "external_links.pkl",
    )

    # load external links
    with open(external_links_data_path, "rb") as f:
        external_links = pickle.load(f)
    print(f"Loaded {len(external_links['dsaids'])} DiSignAtlas metadata (external links)")

    # get DOID for each dsa
    dsaids_2_doids = {}
    for dsaid, external_link in zip(
        external_links["dsaids"], external_links["external_links"]
    ):
        do_ids = [
            link.replace("DO:", "DOID:") for link in external_link if link.startswith("DO")
        ]
        if len(do_ids) > 0:
            dsaids_2_doids[dsaid] = do_ids

    return dsaids_2_doids

def get_clean_set_dsaids(library_strategy: Optional[List[str]] = ["RNA-Seq", "Microarray"]) -> set:
    """
    Get clean set of DSAIDs - Human, RNA-seq & with info
    """
    # variables
    pct_thr = 0.5
    
    df_data_info = load_dsa_info()
    print(f"Loaded Nº DSAIDs: {len(df_data_info)}")

    # Filter Human species
    df_data_info = df_data_info[df_data_info["organism"] == "Homo sapiens"]
    print(f"Filter Human Species Nº DSAIDs: {len(df_data_info)}")

    # Filter RNA-seq & Microarray
    df_data_info = df_data_info[df_data_info["library_strategy"].isin(library_strategy)]
    print(f"Filter {library_strategy} Nº DSAIDs: {len(df_data_info)}")

    # Filter DSAIDs with info
    _processed_dsaids = get_processed_ids()
    df_data_info = df_data_info[df_data_info["dsaid"].isin(_processed_dsaids)]
    print(f"Filter DSAIDs with info Nº DSAIDs: {len(df_data_info)}")

    # Filter to enough Human protein coding genes!
    # Found corner cases of studies with majority mouse genes - we clean these out
    # Load gene signatures
    gene_signatures = load_gene_signatures()
    gene_signatures = {k: v for k, v in gene_signatures.items() if k in df_data_info["dsaid"].unique()}
    print(f"Loaded signatures Nº DSAIDs: {len(gene_signatures)}")

    # Load human protein coding genes
    human_protein_coding_genes = get_human_entrez_protein_coding_ids()
    print(f"Loaded human protein coding genes: {len(human_protein_coding_genes)}")

    # Calculate the percentage of human protein coding genes in each signature    
    dsaids_to_remove = set()
    for dsaid_i, s_i in gene_signatures.items():
        _pct_human_coding_genes = len(set(s_i["gene_ids"]) & set(human_protein_coding_genes)) / len(s_i["gene_ids"])
        if _pct_human_coding_genes < pct_thr:
            dsaids_to_remove.add(dsaid_i)
    
    gene_signatures = {k: v for k, v in gene_signatures.items() if k not in dsaids_to_remove}
    df_data_info = df_data_info[df_data_info["dsaid"].isin(gene_signatures.keys())]
    print(f"Filter to enough Human protein coding genes Nº DSAIDs: {len(df_data_info)}")
    return set(df_data_info["dsaid"].unique())

def get_doids_with_umls(library_strategy: Optional[List[str]] = ["RNA-Seq", "Microarray"])-> List:
    """
    Get a set of DO IDs that have UMLS codes associated with them.
    """

    # first get clean set of DSAIDs
    clean_dsaids = get_clean_set_dsaids(library_strategy)

    # load DO graph
    do_graph = load_do_graph()

    # get mapping from DSAID to DOID    
    umls_2_doid = get_umls_2_doid_mapping(do_graph)
    df_data_info = load_dsa_info()

    # filter by clean dsaids
    df_data_info = df_data_info[df_data_info["dsaid"].isin(clean_dsaids)]
    print(f"Filter by clean DSAIDs Nº DSAIDs: {len(df_data_info)}")

    # filter by presence of UMLS codes
    df_data_info.dropna(subset=["diseaseid"], inplace=True)
    print(f"Filter by presence of UMLS codes Nº DSAIDs: {len(df_data_info)}")

    # filter by mapping from UMLS to DOID
    df_data_info = df_data_info[df_data_info["diseaseid"].isin(umls_2_doid.keys())]
    print(f"Filter by mapping from UMLS to DOID Nº DSAIDs: {len(df_data_info)}")

    return df_data_info["dsaid"].unique().tolist()

def generate_output_folder_dir():
    # Step 1: Generate today's date string
    today = datetime.now().strftime("%y-%m-%d")

    # Step 2: Define the base output directory
    # base_output_dir = os.path.join("..", "outputs")
    base_output_dir = "/aloy/home/ddalton/projects/scGPT_playground/outputs"

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

def load_df_do_pairs()->pd.DataFrame:
    """Load Disease Pairs"""
    similarity_file = "/aloy/home/ddalton/projects/BQ_diseases/outputs/do_similarity_results.pkl"
    results = pickle.load(open(similarity_file, "rb"))

    return pd.DataFrame(
        results,
        columns=[
            "do1",
            "do2",
            "pair_sorted",
            "resnik",
            "lin",
            "jiang",
            "mica_node",
            "shortest_path_length"
        ]
    )
