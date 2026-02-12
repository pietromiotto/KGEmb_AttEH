# -*- coding: utf-8 -*-
"""
Mixture Models – Combining Hyperbolic and Euclidean Models
"""
import torch
from torch import nn
from models.hyperbolic import AttH
from models.euclidean import AttE
from models.base import KGModel

MIXTURE_MODELS = ["AttEH"]

class AttEH(KGModel):
    """
    Fully-gated mixture of AttH and AttE.
    
    score(h,r,t) = [σ_r·sim_H + (1-σ_r)·sim_E] 
                 + [σ_r·b_h_H + (1-σ_r)·b_h_E] 
                 + [σ_r·b_t_H + (1-σ_r)·b_t_E]

    The gate σ_r is computed once in get_queries and passed via
    the lhs_e tuple to the score and similarity_score methods.
    
    This class overrides `get_rhs`, `similarity_score`, and `score`
    from the base KGModel to implement this mixing logic.
    """

    def __init__(self, args, disease_ids=None, restrict_to_diseases=False):
        super().__init__(args.sizes, args.rank, args.dropout,
                         args.gamma, args.dtype, args.bias, args.init_size)

        # Initialize disease-related attributes
        self.disease_ids = disease_ids
        self.restrict_to_diseases = restrict_to_diseases

        # Initialize component models
        self.head_H = AttH(args, disease_ids, restrict_to_diseases)
        self.head_E = AttE(args, disease_ids, restrict_to_diseases)

        # Gate to control the mixture ratio
        self.gate = nn.Embedding(args.sizes[1], 1)
        nn.init.zeros_(self.gate.weight)  # σ ≈ 0.5 at start


    def get_queries(self, queries):
        """
        Computes mixed head bias and packs query embeddings + gate (σ).
        
        σ is computed ONCE here and passed along.
        """
        (lhs_H, c), bias_H = self.head_H.get_queries(queries)  # hyperbolic
        lhs_E,      bias_E = self.head_E.get_queries(queries)  # euclidean

        # Compute the gate value per relation on the queries
        σ = torch.sigmoid(self.gate(queries[:, 1]))  # [B, 1]

        lhs_pack = (lhs_H, lhs_E, c, σ)
        
        # mixed eucl and hyperb head bias
        # TODO: avoid extra gate computations if bias is 'none' or 'constant'
        mixed_head_bias = σ * bias_H + (1.0 - σ) * bias_E

        return lhs_pack, mixed_head_bias

    def get_rhs(self, queries, eval_mode):
        """
        Returns unmixed tail embeddings and unmixed
        tail biases from BOTH models.
        """
        # get Hyperbolic tails
        rhs_H, bias_t_H = self.head_H.get_rhs(queries, eval_mode)
        
        # get Euclidean tails
        rhs_E, bias_t_E = self.head_E.get_rhs(queries, eval_mode)
        
        rhs_pack = (rhs_H, rhs_E)
        bias_pack = (bias_t_H, bias_t_E)
        
        return rhs_pack, bias_pack

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """
        Unpacks LHS pack (with σ) and RHS pack.
        Uses σ to compute the mixed similarity score.
        """
        
        lhs_H, lhs_E, c, σ = lhs_e

        
        rhs_H, rhs_E = rhs_e  # Unpack the tuple from our new get_rhs
        
        # compute ATTH and ATTE score 
        score_H = self.head_H.similarity_score((lhs_H, c), rhs_H, eval_mode)
        score_E = self.head_E.similarity_score(lhs_E, rhs_E, eval_mode)

        # mix the similarity scores
        return σ * score_H + (1.0 - σ) * score_E

    def score(self, lhs, rhs, eval_mode):
        """
        OVERRIDDEN from KGModel.
        This method assembles all mixed components.
        - Gets σ from lhs_e (packed by get_queries).
        - Gets unmixed tail biases from rhs_biases (packed by get_rhs).
        - Computes the final mixed tail bias here.
        """
        lhs_e, lhs_biases = lhs  # lhs_biases is the mixed head bias
        rhs_e, rhs_biases = rhs  # rhs_biases = (bias_t_H, bias_t_E)

        # compute mixed similarity score
        sim_score = self.similarity_score(lhs_e, rhs_e, eval_mode)

        σ = lhs_e[3] #lhs_e = (lhs_H, lhs_E, c, σ)

        bias_t_H, bias_t_E = rhs_biases # rhs_biases = (bias_t_H, bias_t_E)
        
        # mix the tail biases
        if eval_mode:
            # bias_t_H/E shape: [N, 1], σ shape: [B, 1]
            # We need [B, N]
            # (B, 1) * (1, N) + (B, 1) * (1, N) -> (B, N)
            mixed_tail_bias = σ * bias_t_H.t() + (1.0 - σ) * bias_t_E.t()
        else:
            # bias_t_H/E shape: [B, 1], σ shape: [B, 1]
            mixed_tail_bias = σ * bias_t_H + (1.0 - σ) * bias_t_E
        

        if self.bias == 'constant':
            return self.gamma.item() + sim_score
        elif self.bias == 'learn':
            # lhs_biases = mixed head bias [B, 1]
            # mixed_tail_bias is [B, 1] or [B, N] (if eval_mode)
            # sim_score is mixed similarity [B, 1] or [B, N]
            return lhs_biases + mixed_tail_bias + sim_score
        else:
            # self.bias == 'none'
            return sim_score

    def get_factors(self, queries):
        """
        Unchanged from original.
        Provides factors for regularization.
        """
        hH, rH, tH = self.head_H.get_factors(queries)
        hE, rE, tE = self.head_E.get_factors(queries)

        # Concatenate to avoid shape mismatch
        return (
            torch.cat([hH, hE], dim=1),
            torch.cat([rH, rE], dim=1),
            torch.cat([tH, tE], dim=1),
        )

    # NOTE: We do NOT need to override `forward` or `get_ranking`.
    # The base KGModel methods will call our new `get_queries`,
    # `get_rhs`, and `score` methods, which now contain all
    # the correct mixing logic.
