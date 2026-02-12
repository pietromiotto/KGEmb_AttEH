import argparse
import json
import os
import pickle
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F

def find_relation_id(rel2idx, relation_name):
    """
    Find relation ID case-insensitively and ignoring spaces.
    """
    relation_name_processed = relation_name.lower().replace(" ", "")
    for name, idx in rel2idx.items():
        if name.lower().replace(" ", "") == relation_name_processed:
            return idx
    return None

def main():
    parser = argparse.ArgumentParser(description="Get curvature for a specific relation from a trained model.")
    parser.add_argument('--model_dir', type=str, required=True, help='Directory where the model and config are stored.')
    parser.add_argument('--relation', type=str, help='The name of the relation to get the curvature for.')
    parser.add_argument('--rel2idx_path', type=str, default=None, help='Path to rel2idx.pickle file (optional).')
    parser.add_argument('--all', action='store_true', help='Get curvatures for all relations.')
    args = parser.parse_args()

    if not args.all and not args.relation:
        parser.error('Either --relation or --all must be specified.')

    model_dir = Path(args.model_dir)
    config_path = model_dir / 'config.json'
    model_path = model_dir / 'model.pt'

    if not model_dir.is_dir() or not config_path.exists() or not model_path.exists():
        print(f"Error: Model directory '{args.model_dir}' is not valid.")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    dataset = config.get('dataset')
    if not dataset:
        print("Error: Dataset not found in config.json. Please specify --rel2idx_path.")
        return

    if args.rel2idx_path:
        rel2idx_path = Path(args.rel2idx_path)
    else:
        # N.W. we assume data/{dataset}/rel2idx.pickle
        rel2idx_path = Path('data') / dataset / 'rel2idx.pickle'

    if not rel2idx_path.exists():
        print(f"Error: rel2idx file not found at '{rel2idx_path}'.")
        return

    train_path = rel2idx_path.parent / 'train.pickle'
    train_triples = None
    if args.all:
        if not train_path.exists():
            print(f"Warning: 'train.pickle' not found at '{train_path}'. Cannot calculate relation frequencies.")
        else:
            with open(train_path, 'rb') as f:
                train_triples = pickle.load(f)

    with open(rel2idx_path, 'rb') as f:
        rel2idx = pickle.load(f)

    model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    
    raw_c = model_state_dict.get('c')

    if raw_c is None:
        print("Curvature parameter 'c' not found in the model.")
        return

    # Apply softplus to get the actual curvature, as done in the model
    curvatures = F.softplus(raw_c)

    if args.all:
        relation_counts = None
        total_triples = 0
        if train_triples is not None:
            relation_ids_in_train = [triple[1] for triple in train_triples]
            relation_counts = Counter(relation_ids_in_train)
            total_triples = len(train_triples)

        print("All relation curvatures, sorted from most to least hierarchical:")
        all_curvatures = []
        is_single_curvature = curvatures.numel() == 1
        for rel_name, rel_id in rel2idx.items():
            if is_single_curvature:
                curvature = curvatures.item()
            else:
                if rel_id < len(curvatures):
                    curvature = curvatures[rel_id].item()
                else:
                    print(f"Warning: Relation ID {rel_id} for '{rel_name}' is out of bounds for curvature tensor.")
                    continue
            
            percentage = 0.0
            if relation_counts and total_triples > 0:
                count = relation_counts.get(rel_id, 0)
                percentage = (count / total_triples) * 100

            all_curvatures.append((rel_name, curvature, percentage))
        
        all_curvatures.sort(key=lambda x: x[1], reverse=True)

        for rel_name, curvature, percentage in all_curvatures:
            if train_triples is not None:
                print(f"Curvature for '{rel_name}' ({percentage:.2f}%): {curvature}")
            else:
                print(f"Curvature for '{rel_name}': {curvature}")

    else:
        relation_name = args.relation
        relation_id = find_relation_id(rel2idx, relation_name)

        if relation_id is None:
            print(f"Relation '{relation_name}' not found.")
            return

        if curvatures.numel() == 1:
            
            curvature = curvatures.item()
            print(f"Model has a single curvature for all relations: {curvature}")
        else:
            
            if relation_id < len(curvatures):
                curvature = curvatures[relation_id].item()
                print(f"Curvature for relation '{relation_name}' (ID: {relation_id}): {curvature}")
            else:
                print(f"Error: Relation ID {relation_id} is out of bounds for the curvature tensor.")

if __name__ == '__main__':
    main()
