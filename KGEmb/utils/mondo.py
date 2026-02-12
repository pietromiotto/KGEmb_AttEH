"""
Helper functions for handling MONDO ontology interactions, including splitting
test sets by the official rare disease subset and by ontological depth.
"""
import re
import obonet
import torch
import networkx as nx

def parse_mondo_id(uri: str) -> str | None:
    """Parses a MONDO ID from a full URI."""
    if not isinstance(uri, str):
        return None
    match = re.search(r'MONDO_(\d+)', uri)
    if match:
        return f"MONDO:{match.group(1)}"
    return None

# --- Official Rare Subset Logic ---
def split_by_rare_subset(test_examples, idx2ent, subset_obo_path="mondo-rare.obo"):
    """
    Splits test examples into 'rare' and 'non-rare' based on the official MONDO subset file.
    """
    print(f"\nSplitting test set using official rare disease subset from {subset_obo_path}")
    try:
        rare_graph = obonet.read_obo(subset_obo_path)
        official_rare_ids = set(rare_graph.nodes())
        print(f"Found {len(official_rare_ids)} diseases in the official rare subset.")
    except FileNotFoundError:
        print(f"FATAL ERROR: '{subset_obo_path}' not found. Please run: wget http://purl.obolibrary.org/obo/mondo/subsets/mondo-rare.obo")
        exit()
    #create a mask for rare diseases inside the test set
    is_rare_mask = torch.tensor([
        (parse_mondo_id(idx2ent.get(int(ex[2]))) in official_rare_ids) for ex in test_examples
    ], dtype=torch.bool)

    rare = test_examples[is_rare_mask]
    non_rare = test_examples[~is_rare_mask]
    print(f"Official Rare Disease examples: {len(rare)} | Non-Rare Disease examples: {len(non_rare)}")
    return rare, non_rare

