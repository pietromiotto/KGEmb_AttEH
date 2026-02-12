import argparse
import json
import pickle
from pathlib import Path
import torch
import numpy as np
import warnings

# --- SUPPRESS WARNINGS ---
warnings.filterwarnings("ignore", category=FutureWarning) 

def find_relation_id(rel2idx, relation_name):
    """
    Find relation ID case-insensitively and ignoring spaces.
    """
    relation_name_processed = relation_name.lower().replace(" ", "")
    for name, idx in rel2idx.items():
        if name.lower().replace(" ", "") == relation_name_processed:
            return idx
    return None

def get_five_num_summary(tensor):
    """
    Calculates Mean, Std, and the 5-number summary (Min, Q1, Med, Q3, Max)
    necessary for boxplots.
    """
    return {
        'mean':   torch.mean(tensor).item(),
        'std':    torch.std(tensor).item(),
        'min':    torch.min(tensor).item(),
        'q1':     torch.quantile(tensor, 0.25).item(), # 25th percentile (Box bottom)
        'median': torch.quantile(tensor, 0.50).item(), # 50th percentile (Box line)
        'q3':     torch.quantile(tensor, 0.75).item(), # 75th percentile (Box top)
        'max':    torch.max(tensor).item()
    }

def main():
    parser = argparse.ArgumentParser(description="Get gate values (sigma) for the AttEH model.")
    parser.add_argument('--model_dir', type=str, required=True, help='Directory where the model and config are stored.')
    parser.add_argument('--relation', type=str, default="Has Disease", help='The name of the relation to extract the gate for (default: "Has Disease").')
    parser.add_argument('--rel2idx_path', type=str, default=None, help='Path to rel2idx.pickle file (optional).')
    parser.add_argument('--all', action='store_true', help='Get gate values for all relations.')
    parser.add_argument('--average', action='store_true', help='Compute full boxplot statistics (requires --all).')
    
    args = parser.parse_args()

    if args.average and not args.all:
        parser.error("--average requires --all to be specified.")

    model_dir = Path(args.model_dir)
    config_path = model_dir / 'config.json'
    model_path = model_dir / 'model.pt'

    
    if not model_dir.is_dir() or not config_path.exists() or not model_path.exists():
        print(f"Error: Model directory '{args.model_dir}' is not valid or missing files.")
        return

    
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_rank = config.get('rank', 'Unknown') #model_rank='Unknown' if 'rank' not in config 

    if args.rel2idx_path:
        rel2idx_path = Path(args.rel2idx_path)
    else:
        dataset = config.get('dataset')
        if not dataset:
            print("Error: Dataset not found in config.json. Please specify --rel2idx_path.")
            return
        rel2idx_path = Path('data') / dataset / 'rel2idx.pickle'

    if not rel2idx_path.exists():
        print(f"Error: rel2idx file not found at '{rel2idx_path}'.")
        return

   
    with open(rel2idx_path, 'rb') as f:
        rel2idx = pickle.load(f)

    # load Model
    try:
        model_state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    except TypeError:
        model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # extract Gate
    raw_gate_weights = model_state_dict.get('gate.weight')

    if raw_gate_weights is None:
        print("Error: Parameter 'gate.weight' not found.")
        return

    gate_values = torch.sigmoid(raw_gate_weights)

    
    if args.all:
        valid_sigmas = []
        all_gates_list = []

        for rel_name, rel_id in rel2idx.items():
            if rel_id < len(gate_values):
                val = gate_values[rel_id].item()
                valid_sigmas.append(val)
                all_gates_list.append((rel_name, val))
        
        all_gates_list.sort(key=lambda x: x[1], reverse=True)

        if args.average:
            # --- STATISTICS FOR BOXPLOTS ---
            hyp_tensor = torch.tensor(valid_sigmas)          # σ
            euc_tensor = 1.0 - hyp_tensor                    # 1 - σ
            
            h_stats = get_five_num_summary(hyp_tensor)
            e_stats = get_five_num_summary(euc_tensor)

            print("\n" + "="*85)
            print(f"  MODEL REPORT  |  RANK: {model_rank}")
            print("="*85)
            print(f"  Boxplot Statistics (N={len(valid_sigmas)} relations)")
            print("-" * 85)
            
            # Header
            header = f"  {'METRIC':<15} | {'MEAN':<8} {'STD':<8} | {'MIN':<8} {'Q1(25%)':<8} {'MED(50%)':<8} {'Q3(75%)':<8} {'MAX':<8}"
            print(header)
            print("-" * 85)
            
            # Rows
            def print_row(name, s):
                print(f"  {name:<15} | {s['mean']:.4f}   {s['std']:.4f}   | {s['min']:.4f}   {s['q1']:.4f}   {s['median']:.4f}   {s['q3']:.4f}   {s['max']:.4f}")

            print_row("Hyperbolic (σ)", h_stats)
            print_row("Euclidean", e_stats)
            
            print("-" * 85)
            print("  * Q1/Q3 define the box edges. Median is the center line. Min/Max are whiskers.")
            print("="*85 + "\n")
        
        else:
            print(f"\nAll relation gate values (Rank: {model_rank})")
            print(f"{'Relation Name':<40} | {'Gate (σ)':<10} | {'Interpretation'}")
            print("-" * 75)
            for rel_name, val in all_gates_list:
                interp = "Hyperbolic" if val > 0.5 else "Euclidean"
                print(f"{rel_name:<40} | {val:.4f}     | {interp}")

    else:
        # Single relation
        relation_name = args.relation
        relation_id = find_relation_id(rel2idx, relation_name)

        if relation_id is None:
            print(f"Error: Relation '{relation_name}' not found.")
            return

        if relation_id >= len(gate_values):
            print(f"Error: Relation ID {relation_id} is out of bounds.")
            return

        sigma = gate_values[relation_id].item()
        
        print(f"\nModel Rank: {model_rank}")
        print(f"Relation:   '{relation_name}' (ID: {relation_id})")
        print("-" * 30)
        print(f"Gate Value (σ):    {sigma:.6f}")
        print(f"Hyperbolic Weight: {sigma:.2%}")
        print(f"Euclidean Weight:  {1.0 - sigma:.2%}")

if __name__ == '__main__':
    main()
