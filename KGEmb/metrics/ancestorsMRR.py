#SCRIPT TO COMPUTE ANCESTORS-BASED MRR (WITH JACCARD PENALTY) FOR ANY MODEL

import re, json, pickle, torch, numpy as np
import scipy.sparse as sp
from typing import Dict, Set, List, Tuple
from tqdm import tqdm
import time

MONDO_RE = re.compile(r"MONDO_\d+")

def uri_to_mondo(uri: str) -> str | None:
    """Extract 'MONDO_XXXXXXXX' from a URI; return None if not present."""
    m = MONDO_RE.search(uri)
    return m.group(0) if m else None


def build_idx2mondo(ent2idx_path: str) -> Dict[int, str | None]:
    """
    Load the `ent2idx.pickle` produced by KGDataset and map every **entity index**
    to its MONDO identifier (or None if the entity is not a disease node).
    """
    print(f"Building index-to-MONDO mapping from {ent2idx_path}...")
    with open(ent2idx_path, "rb") as f:
        ent2idx: Dict[str, int] = pickle.load(f)

    idx2mondo: Dict[int, str | None] = {}
    for uri, idx in ent2idx.items():
        idx2mondo[idx] = uri_to_mondo(uri)    # may be None
    print("Index-to-MONDO mapping built.")
    return idx2mondo

def build_ancestor_matrix(
        idx2mondo: Dict[int, str | None],
        ancestors_json: str
) -> tuple[sp.csr_matrix, Dict[int, int]]:
    """
    Returns
    -------
    anc_mat : CSR matrix  (n_entities × n_unique_mondos)  with 0/1 entries
    idx_map : dict index→row  (needed later because some entities have no MONDO ID)
    """
    print(f"Loading ancestor data from {ancestors_json}...")
    with open(ancestors_json) as f:
        mondo2anc: Dict[str, List[str]] = json.load(f)

    #replace ':' with '_'
    print("Normalizing ancestor data (replacing ':' with '_')...")
    mondo2anc_normalized: Dict[str, List[str]] = {}
    for key, ancestors in mondo2anc.items():
        norm_key = key.replace(":", "_")
        norm_ancestors = [a.replace(":", "_") for a in ancestors]
        mondo2anc_normalized[norm_key] = norm_ancestors

    print("Constructing sparse ancestor matrix...")
    start_time = time.time()
    
    mondo_ids_in_kg = {m for m in idx2mondo.values() if m}
    
    all_ancestors_from_json = {
        a for m in mondo_ids_in_kg 
        for a in mondo2anc_normalized.get(m, [])
    }
    
    all_cols    = sorted(all_ancestors_from_json | mondo_ids_in_kg)
    col_index   = {m: j for j, m in enumerate(all_cols)}

    rows, cols = [], []
    for idx, mondo in tqdm(idx2mondo.items(), desc="Populating matrix rows"):
        if not mondo:        # non-disease entity → leave row empty
            continue
        
        # Use normalized data to get ancestor list
        anc_list = mondo2anc_normalized.get(mondo, [])
        
        for m in anc_list + [mondo]:      # include itself
            # Check if the ancestor is in our columns
            if m in col_index:
                rows.append(idx)
                cols.append(col_index[m])

    data = np.ones(len(rows), dtype=np.uint8)
    
    # Ensures shape is valid even if all_cols is empty (though it shouldn't be)
    shape_cols = len(all_cols) if len(all_cols) > 0 else 1
    shape = (max(idx2mondo) + 1, shape_cols)
        
    mat  = sp.csr_matrix((data, (rows, cols)), shape=shape, dtype=np.uint8)
    
    end_time = time.time()
    print(f"Sparse ancestor matrix built in {end_time - start_time:.2f} seconds. Shape: {mat.shape}, NNZ: {mat.nnz}")
    return mat, col_index


def _vectorized_weighted_rank(
        scores: torch.Tensor,
        target_score: float,
        target_idx: int,
        anc_mat: sp.csr_matrix,
) -> float:
    """
    Compute the 'soft-penalty' rank for a single query using vectorized operations.
    """
    # Find all predictions with a score >= to the *pre-computed* target's score
    higher_indices_tensor = (scores >= target_score).nonzero(as_tuple=True)[0]
    higher_indices = higher_indices_tensor.numpy()

    # Get the sparse ancestor vectors
    target_row = anc_mat.getrow(target_idx)
    
    # If the target itself is not a disease, it has no ancestors.

    if target_row.nnz == 0:
        return float(len(higher_indices))

    higher_rows = anc_mat[higher_indices]

    # vectorized computation of Jaccard similarity
    intersection = higher_rows.dot(target_row.T).toarray().ravel()
    
    # get non-zero counts for each row in the sliced matrix
    higher_nnz = np.diff(higher_rows.indptr)
    
    union = higher_nnz + target_row.nnz - intersection
    
    # avoid division by zero
    jaccard_sims = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union!=0)
    
    penalties = 1.0 - jaccard_sims
    
    # The rank is 1 + the sum of penalties for all OTHER higher-scoring items.
    # We sum all penalties and subtract the target's own penalty (which should be 0).
    total_penalty = np.sum(penalties)

    target_in_higher = (higher_indices == target_idx)
    if np.any(target_in_higher):
        target_penalty = penalties[target_in_higher].item()
    else:
        target_penalty = 0.0
    
    return 1.0 + total_penalty - target_penalty


def _batch_weighted_ranks(
        model,
        queries: torch.Tensor,
        filter_dict,
        anc_mat: sp.csr_matrix,
        side: str,
        batch_size: int = 512
) -> torch.Tensor:
    """
    Mirror of KGModel.get_ranking, but returns **weighted** ranks (float32).
    """
    wranks = torch.ones(len(queries), dtype=torch.float32)
    num_relations = model.sizes[1] // 2
    with torch.no_grad():
        b = 0
        print("Pre-computing all candidate embeddings (once per side)...")
        candidates = model.get_rhs(queries, eval_mode=True)
        print("Candidate embeddings computed. Starting batch evaluation...")
        
        with tqdm(total=len(queries), desc=f"Weighted MRR ({side})") as pbar:
            while b < len(queries):
                q_batch = queries[b: b + batch_size].to(model.entity.weight.device)
                q_emb   = model.get_queries(q_batch)

                # get scores against all candidates (eval_mode=True)
                scores  = model.score(q_emb, candidates, eval_mode=True)

                # get scores for *just* the targets (eval_mode=False)
                rhs_emb, rhs_biases = model.get_rhs(q_batch, eval_mode=False)
                target_scores_vec = model.score(q_emb, (rhs_emb, rhs_biases), eval_mode=False)

                for i, q in enumerate(q_batch):
                    head, rel, target_idx = q[0].item(), q[1].item(), q[2].item()
                    
                    filter_key = None
                    if side == 'rhs':
                        filter_key = (head, rel)
                    else:  # side == 'lhs'
                        original_rel = rel - num_relations
                        filter_key = (head, original_rel)

                    all_true_answers = filter_dict.get(filter_key, [])
                    filt = [ans for ans in all_true_answers if ans != target_idx]
                    
                    if filt:
                        scores[i, torch.as_tensor(filt, device=scores.device)] = -1e6

                # disease masking (if enabled in the model)
                if getattr(model, "restrict_to_diseases", False) and hasattr(model, "disease_mask"):
                    non_dis = (~model.disease_mask).nonzero(as_tuple=True)[0]
                    scores[:, non_dis.to(scores.device)] = -1e6

                scores_cpu = scores.cpu()
                target_idxs = q_batch[:, 2].cpu()
                target_scores_vec_cpu = target_scores_vec.cpu()

                # vectorized rank computation for each query in the batch
                for i in range(scores_cpu.size(0)):
                    wranks[b + i] = _vectorized_weighted_rank(
                        scores_cpu[i],
                        target_scores_vec_cpu[i].item(),
                        target_idxs[i].item(),
                        anc_mat
                    )
                
                b += batch_size
                pbar.update(q_batch.size(0))

    return wranks


def compute_weighted_mrr(
        model,
        examples: torch.Tensor,
        filters,
        anc_mat: sp.csr_matrix,
        batch_size: int = 512,
        sides: List[str] = ['rhs']
) -> Dict[str, float]:
    """
    Exactly the same API pattern as `KGModel.compute_metrics`, but returns
    only the MRR variant. By default, only computes for the 'rhs' side.
    
    Args:
        ...
        sides (List[str]): A list of sides to evaluate, e.g., ['rhs', 'lhs']. 
                            Defaults to ['rhs'].
    """
    out = {}
    for side in sides:
        if side not in ['rhs', 'lhs']:
            raise ValueError(f"side must be 'rhs' or 'lhs', but got {side}")

        print(f"\n--- Computing weighted MRR for '{side}' side ---")
        q = examples.clone()
        if side == "lhs":
            q[:, [0, 2]] = q[:, [2, 0]]     # swap head & tail
            q[:, 1] += model.sizes[1] // 2   # inverse relation

        wranks = _batch_weighted_ranks(
            model, q, filters[side], anc_mat, side, batch_size
        )
        out[side] = torch.mean(1.0 / wranks).item()
    return out
