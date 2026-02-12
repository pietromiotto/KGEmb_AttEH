"""
Comprehensive evaluation script for knowledge graph models.
Supports splitting the test set by:
1. Disease frequency in the dataset (from test set).
2. The official MONDO rare disease subset.
"""

import argparse
import json
import os
import pandas as pd
import pickle
import numpy as np
from collections import Counter

import torch

import models
from datasets.kg_dataset import KGDataset

from utils.train import avg_both, format_metrics, avg_both_rhs
from utils.mondo import split_by_rare_subset

from metrics import ancestorsMRR as AncestorMetrics


parser = argparse.ArgumentParser(description="Test")
parser.add_argument(
    '--model_dir',
    required=True,
    help="Model path"
)
parser.add_argument(
    '--split_by_frequency',
    action='store_true',
    help="Split test examples by disease frequency"
)
parser.add_argument(
    '--split_by_official_rare',
    action='store_true',
    help="Split by the official MONDO rare disease subset (mondo-rare.obo)."
)
parser.add_argument(
    '--all',
    action='store_true',
    help="Run all three evaluation modes (full, by frequency, by official rare) sequentially."
)
parser.add_argument(
    '--ancestors_path',
    type=str,
    default='metrics/mondo_ancestors.json',
    help="Path to the mondo_ancestors.json file."
)
parser.add_argument(
    '--rare_subset_path',
    type=str,
    default='mondo-rare.obo',
    help="Path to the mondo-rare.obo subset file."
)

############################ HELPERS ###############################################
def split_by_disease_frequency(test_examples, min_count=50):
    """Splits test examples into frequent and rare based on tail entity count."""
    print(f"Splitting {len(test_examples)} examples by frequency (min_count={min_count})...")
    
    # Convert tensor to numpy for easier processing if necessary
    if isinstance(test_examples, torch.Tensor):
        examples_array = test_examples.cpu().numpy()
    else:
        examples_array = test_examples
    
    # Count disease occurrences 
    # N.W. We assume frequency of a disease is equal to #patients affected by it
    disease_counts = Counter(int(t[2]) for t in examples_array)
    
    frequent_diseases = {disease for disease, count in disease_counts.items() 
                         if count >= min_count}
    
    # Create masks for frequent and rare diseases
    frequent_mask = torch.tensor([int(triple[2]) in frequent_diseases for triple in examples_array], dtype=torch.bool)
    rare_mask = ~frequent_mask
    
    # Use masks to split the original tensor
    frequent = test_examples[frequent_mask]
    rare = test_examples[rare_mask]
    
    print(f"Frequent disease examples: {len(frequent)}")
    print(f"Rare disease examples: {len(rare)}")
    
    return frequent, rare


def print_indented(text, indent="  "):
    """Prints text with a given indentation."""
    for line in str(text).splitlines():
        print(f"{indent}{line}")


def evaluate_subset(model, model_restricted, examples, filters, anc_mat, title, split_name):
    """
    Runs a full evaluation suite on a subset of examples and prints formatted results.
    """
    header_line = f"--- {title} ---"
    print(f"\n{header_line}")
    print(f"Evaluating on {len(examples)} examples...")
    
    if len(examples) == 0:
        print("Skipping evaluation: No examples in this subset.")
        print(f"{'-' * len(header_line)}\n")
        return

    try:
        # Compute raw metrics once
        raw_unrestricted = model.compute_metrics(examples, filters)
        raw_restricted = model_restricted.compute_metrics(examples, filters)

        # Aggregate metrics
        metrics_full = avg_both(*raw_unrestricted)
        metrics_rhs = avg_both_rhs(*raw_unrestricted)
        metrics_restricted_rhs = avg_both_rhs(*raw_restricted)

        print("\n  Full-KG Evaluation (LHS + RHS average):")
        print_indented(format_metrics(metrics_full, split=split_name), indent="    ")
        
        print("\n  Full-KG Evaluation (RHS-only):")
        print_indented(format_metrics(metrics_rhs, split=split_name), indent="    ")
        
        print("\n  Disease-only Evaluation (RHS-only):")
        print_indented(format_metrics(metrics_restricted_rhs, split=split_name), indent="    ")

        # WeightedMRR
        w_mrr = AncestorMetrics.compute_weighted_mrr(
            model_restricted, examples, filters, anc_mat, 
            batch_size=256, sides=['rhs']
        )
        print("\n  Weighted-by-Ancestors MRR (RHS-only):")
        print(f"    RHS : {w_mrr['rhs']:.4f}")
    
    except Exception as e:
        print(f"\n  [ERROR] An error occurred during evaluation for '{title}': {e}")
    
    print(f"{'-' * len(header_line)}\n")


######################### MAIN TEST FUNCTION ###############################
def test(args):
    
    print("\n--- 1. LOADING CONFIGURATION AND DATA ---")
    
    with open(os.path.join(args.model_dir, "config.json"), "r") as f:
        config = json.load(f)
    config.update(vars(args)) # Runtime args override config
    args = argparse.Namespace(**config)
    print(f"Loaded config from {args.model_dir}")

    # create dataset
    dataset_path = os.path.join(os.environ["DATA_PATH"], args.dataset)
    dataset = KGDataset(dataset_path, False)
    test_examples = dataset.get_examples("test")
    print(f"Loaded {len(test_examples)} test examples from {args.dataset}")
    filters = dataset.get_filters()

    # needed for disease-only MRR
    ent2idx_path = f"{dataset_path}/ent2idx.pickle"
    with open(ent2idx_path, "rb") as f:
        ent2idx = pickle.load(f) # Dict[str -> int]

    try:
        nodes = pd.read_csv("KGEmb/nodes_filtered.csv")
    except FileNotFoundError:
        nodes = pd.read_csv("nodes_filtered.csv")
        
    # Filter URIs of type 'disease'
    disease_uris = nodes[nodes["type"] == "Disease"]["name"].tolist()
    # Convert to model indices
    disease_ids = [ent2idx[uri] for uri in disease_uris if uri in ent2idx]
    disease_ids = torch.tensor(disease_ids, dtype=torch.long)
    print(f"Found {len(disease_ids)} disease nodes for restriction.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    
    print("\n--- 2. LOADING MODELS AND ANCESTOR DATA ---")

    # disease-only MRR model
    model_restricted = getattr(models, args.model)(
        args,
        disease_ids=disease_ids,
        restrict_to_diseases=True
    )
    model_restricted.to(device)
    model_restricted.load_state_dict(torch.load(os.path.join(args.model_dir, 'model.pt')))

    # REBUILD DISEASE MASK manually after loading. THIS IS NEEDED DO NOT REMOVE
    if model_restricted.restrict_to_diseases and model_restricted.disease_ids is not None:
        mask = torch.zeros(model_restricted.sizes[0], dtype=torch.bool)
        mask[model_restricted.disease_ids] = True
        model_restricted.register_buffer("disease_mask", mask)
        print(f"Fixed disease mask (restricted model): {mask.sum().item()} disease nodes.")
    
    # normal model
    model = getattr(models, args.model)(
        args,
        disease_ids=disease_ids,
        restrict_to_diseases=False
    )
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model.pt')))
    print(f"Loaded models from {args.model_dir}")
    
    # imported from metrics/ancestorsMRR.py
    idx2mondo = AncestorMetrics.build_idx2mondo(ent2idx_path)
    anc_mat, _ = AncestorMetrics.build_ancestor_matrix(idx2mondo, args.ancestors_path)

    
    print("\n--- 3. PREPARING DATA SPLITS ---")
    
    # --- Split 1: By Frequency (min_count=10) ---
    print("Preparing split by disease frequency (min_count=10)...")
    frequent_split, rare_freq_split = split_by_disease_frequency(test_examples, min_count=10)

    # --- Split 2: By Official Rare Subset ---
    print("\nPreparing split by Official MONDO Rare Disease Subset...")
    idx2ent = {idx: ent for ent, idx in ent2idx.items()}
    print(f"Using rare subset file: {args.rare_subset_path}")
    try:
        rare_official_split, non_rare_official_split = split_by_rare_subset(
            test_examples, 
            idx2ent, 
            args.rare_subset_path
        )
    except FileNotFoundError:
        print(f"[ERROR] Could not find rare subset file: {args.rare_subset_path}")
        print("Skipping 'Official Rare' evaluation.")
        # Set to empty tensors to avoid crashing evaluation if --all is used
        rare_official_split = torch.empty(0, 3, dtype=torch.long, device=device)
        non_rare_official_split = torch.empty(0, 3, dtype=torch.long, device=device)

    
    print("\n--- 4. RUNNING EVALUATION(S) ---")
    
    # --all flag
    if args.all:
        print("Running all evaluation modes...")
        
        
        model_name = config.get('model', 'N/A') 
        rank = config.get('rank', 'N/A')         
        print("\n" + "#"*80)
        print(f"## COMPREHENSIVE REPORT FOR {model_name} at rank {rank} ##")
        print(f"## Model Directory: {args.model_dir} ##")
        print("#"*80)
        
        
        # 1. Full Evaluation
        print("\n" + "#"*80)
        print("EVALUATION: FULL TEST SET")
        print("#"*80)
        evaluate_subset(
            model, model_restricted, test_examples, filters, anc_mat,
            title="FULL TEST SET",
            split_name="test (full)"
        )
        
        # 2. Split by Frequency
        print("\n" + "#"*80)
        print("EVALUATION: SPLIT BY DISEASE FREQUENCY (min_count=10)")
        print("#"*80)
        evaluate_subset(
            model, model_restricted, frequent_split, filters, anc_mat,
            title="FREQUENT DISEASES (by Frequency)",
            split_name="test (freq)"
        )
        evaluate_subset(
            model, model_restricted, rare_freq_split, filters, anc_mat,
            title="RARE DISEASES (by Frequency)",
            split_name="test (rare-freq)"
        )

        # 3. Split by Official Rare
        print("\n" + "#"*80)
        print("EVALUATION: SPLIT BY OFFICIAL MONDO RARE SUBSET")
        print("#"*80)
        evaluate_subset(
            model, model_restricted, non_rare_official_split, filters, anc_mat,
            title="NON-RARE DISEASES (Official)",
            split_name="test (non-rare)"
        )
        evaluate_subset(
            model, model_restricted, rare_official_split, filters, anc_mat,
            title="RARE DISEASES (Official)",
            split_name="test (rare-official)"
        )
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION REPORT COMPLETE")
        print("="*80)

    elif args.split_by_frequency:
        print("Running evaluation: Split by disease frequency (min_count=10)...")
        evaluate_subset(
            model, model_restricted, frequent_split, filters, anc_mat,
            title="FREQUENT DISEASES (by Frequency)",
            split_name="test (freq)"
        )
        evaluate_subset(
            model, model_restricted, rare_freq_split, filters, anc_mat,
            title="RARE DISEASES (by Frequency)",
            split_name="test (rare-freq)"
        )

    elif args.split_by_official_rare:
        print("Running evaluation: Split by Official MONDO Rare Disease Subset...")
        evaluate_subset(
            model, model_restricted, non_rare_official_split, filters, anc_mat,
            title="NON-RARE DISEASES (Official)",
            split_name="test (non-rare)"
        )
        evaluate_subset(
            model, model_restricted, rare_official_split, filters, anc_mat,
            title="RARE DISEASES (Official)",
            split_name="test (rare-official)"
        )
        
    else:
        # Standard evaluation on all test examples
        print("Running evaluation: Full test set...")
        evaluate_subset(
            model, model_restricted, test_examples, filters, anc_mat,
            title="FULL TEST SET",
            split_name="test (full)"
        )
            

if __name__ == "__main__":
    args = parser.parse_args()
    test(args)


