import os
import obonet
import networkx as nx
import math
import datetime
import numpy as np
import scanpy as sc
from typing import *
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

def get_folder_name(base_output_dir:str)->str:
    """Get Folder Name
    Args:
        - output_path (str): Output folder
    Returns:
        - output_dir (str): Output directory
    """
    # Step 1: Generate today's date string
    today = datetime.now().strftime("%y-%m-%d")

    # Step 2: Find the highest existing run number for today
    existing_runs = [
        d for d in os.listdir(base_output_dir)
        if os.path.isdir(os.path.join(base_output_dir, d)) and d.startswith(f"pp_data-{today}")
    ]

    # Extract numbers from existing runs and find the max
    existing_numbers = [
        int(d.split("-")[-1]) for d in existing_runs if d.split("-")[-1].isdigit()
    ]

    # Calculate the next run number
    next_run_number = max(existing_numbers, default=0) + 1

    # Step 3: Create the directory name with zero-padded run number
    output_dir = os.path.join(base_output_dir, f"pp_data-{today}-{next_run_number:02d}")

    # Step 4: Create the directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory created: {output_dir}")
    return output_dir

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

    # Prepare sorted level 1 node list for consistent column order
    nodes = sorted(nodes)

    # For each sample, collect level 1 ancestors
    sample_level1_labels = []
    for doid in adata.obs["do_id"]:

        ancestors = nx.ancestors(do_g, doid)
        ancestors.add(doid) # include itself just in case

        # Keep only those in level 1
        top_level_matches = [n for n in ancestors if n in nodes]
        sample_level1_labels.append(top_level_matches)

    # Fit MultiLabelBinarizer on the fixed level 1 vocabulary
    mlb = MultiLabelBinarizer(classes=nodes)
    Y_multilabel = mlb.fit_transform(sample_level1_labels)

    return Y_multilabel, np.array(nodes)


def clean_multilabel_vectors(Y_multilabel:np.ndarray, nodes:np.ndarray,)->tuple[np.ndarray,np.ndarray]:
    """Clean multilabel vectors by removing nodes with no samples."""
    # Identify nodes with no samples
    node_sums = np.sum(Y_multilabel, axis=0)
    mask_keep = node_sums > 0

    cleaned_Y = Y_multilabel[:, mask_keep]
    cleaned_nodes = np.array(nodes)[mask_keep]

    return cleaned_Y, cleaned_nodes