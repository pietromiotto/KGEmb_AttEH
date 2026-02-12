#SCRIPT TO CREATE ANCESTOR LIST OF EACH DISEASE NODE IN THE KG

import argparse
import json
import pandas as pd
import pickle
from collections import defaultdict
from typing import Set, Dict, List

def load_kg_diseases(nodes_csv_path: str, ent2idx_path: str) -> Set[str]:
    """
    Loads disease MONDO IDs from nodes_filtered.csv and filters them to only
    include diseases that are present in the model's entity index.

    Args:
        nodes_csv_path: Path to the nodes_filtered.csv file.
        ent2idx_path: Path to the ent2idx.pickle file.

    Returns:
        A set of MONDO IDs present as diseases in the KG and known to the model.
    """
    print(f"Loading disease nodes from {nodes_csv_path}...")
    nodes_df = pd.read_csv(nodes_csv_path)
    
    # get nodes of type 'Disease' and extract their URIs
    csv_disease_uris = set(nodes_df[nodes_df['type'] == 'Disease']['name'].astype(str).unique())
    print(f"Found {len(csv_disease_uris)} unique disease URIs in the CSV.")

    # load ent2idx which maps entity URIs to model indices
    print(f"Loading model entity vocabulary from {ent2idx_path}...")
    with open(ent2idx_path, 'rb') as f:
        ent2idx = pickle.load(f)
    model_entities = set(ent2idx.keys())
    print(f"Model knows {len(model_entities)} unique entities.")

    # find the intersection using the original, un-normalized URIs
    relevant_disease_uris = csv_disease_uris.intersection(model_entities)
    print(f"Found {len(relevant_disease_uris)} diseases that are in both the CSV and the model's vocabulary.")
    
    # normalize the relevant URIs to the 'MONDO:xxxx' format for the rest of the script.
    normalized_disease_ids = {
        uri.split('/')[-1].replace('>', '').replace('_', ':') 
        for uri in relevant_disease_uris
    }
    
    return normalized_disease_ids


def parse_mondo_obo(obo_path: str) -> Dict[str, List[str]]:
    """
    Parses the mondo.obo file to create a map of child terms to their direct parents.

    Args:
        obo_path: Path to the mondo.obo file.

    Returns:
        A dictionary mapping each MONDO ID to a list of its direct parent MONDO IDs.
    """
    print(f"Parsing MONDO ontology from {obo_path}...")
    child_to_parents = defaultdict(list)
    current_id = None
    
    with open(obo_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '[Term]':
                current_id = None
            elif line.startswith('id:'):
                current_id = line.split(' ')[1]
            elif line.startswith('is_a:'):
                if current_id:
                    parent_id = line.split(' ')[1]
                    child_to_parents[current_id].append(parent_id)
    
    print(f"Parsed {len(child_to_parents)} terms with parent relationships.")
    return dict(child_to_parents)


def get_all_ancestors(
    child_to_parents_map: Dict[str, List[str]], 
    relevant_diseases: Set[str]
) -> Dict[str, List[str]]:
    """
    For each relevant disease, finds all of its ancestors (transitive closure).

    Args:
        child_to_parents_map: A map of a child to its direct parents.
        relevant_diseases: A set of MONDO IDs to compute ancestors for.

    Returns:
        A dictionary mapping each relevant MONDO ID to a list of all its ancestor MONDO IDs.
    """
    print("Building full ancestor map for relevant diseases...")
    ancestor_map = {}
    
    # Memo cache to store results for already computed ancestors
    memo = {}

    for i, disease_id in enumerate(list(relevant_diseases)):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(relevant_diseases)} diseases...")

        if disease_id in memo:
            continue

        # stack for iterative depth-first traversal
        nodes_to_visit = list(child_to_parents_map.get(disease_id, []))
        all_ancestors_for_id = set(nodes_to_visit)
        
        processed_nodes = {disease_id}

        while nodes_to_visit:
            current_node = nodes_to_visit.pop()
            if current_node in processed_nodes:
                continue
            processed_nodes.add(current_node)

            # check if we have already computed ancestors for this node
            if current_node in memo:
                parents = memo[current_node]
            else:
                parents = set(child_to_parents_map.get(current_node, []))
                memo[current_node] = parents

            for parent in parents:
                if parent not in all_ancestors_for_id:
                    all_ancestors_for_id.add(parent)
                    nodes_to_visit.append(parent)
        
        ancestor_map[disease_id] = sorted(list(all_ancestors_for_id))

    print("Finished building ancestor map.")
    return ancestor_map


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Build a JSON file mapping MONDO IDs to their ancestors based on a KG's nodes."
    )
    parser.add_argument(
        '--obo', 
        type=str, 
        required=True, 
        help="Path to the mondo.obo file."
    )
    parser.add_argument(
        '--nodes', 
        type=str, 
        required=True, 
        help="Path to the nodes_filtered.csv file."
    )
    parser.add_argument(
        '--ent2idx',
        type=str,
        required=True,
        help="Path to the ent2idx.pickle file from the model."
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='mondo_ancestors.json', 
        help="Path to save the output JSON file."
    )
    args = parser.parse_args()

    # get the list of diseases present in our knowledge graph AND known to the model
    kg_disease_ids = load_kg_diseases(args.nodes, args.ent2idx)
    
    # parse the entire OBO file to get the complete hierarchy
    child_parent_map = parse_mondo_obo(args.obo)
    
    # compute all ancestors for ONLY the diseases in our KG
    ancestor_map = get_all_ancestors(child_parent_map, kg_disease_ids)
    
    # save the result to a JSON file
    print(f"Saving ancestor map for {len(ancestor_map)} diseases to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(ancestor_map, f, indent=2)
    
    print("Done!")


if __name__ == '__main__':
    main()


