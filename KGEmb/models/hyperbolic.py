"""Hyperbolic Knowledge Graph embedding models where all parameters are defined in tangent spaces."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.base import KGModel
from utils.euclidean import givens_rotations, givens_reflection
from utils.hyperbolic import mobius_add, expmap0, project, hyp_distance_multi_c, hyp_distance_multi_c_elementwise

HYP_MODELS = ["RotH", "RefH", "AttH", "MurP"]


class BaseH(KGModel):
    """Trainable curvature for each relationship."""

    def __init__(self, args, disease_ids=None, restrict_to_diseases=False):
        super(BaseH, self).__init__(
            args.sizes, args.rank, args.dropout, args.gamma,
            args.dtype, args.bias, args.init_size
        )

        self.disease_ids = disease_ids
        self.restrict_to_diseases = restrict_to_diseases

        self.entity.weight.data = self.init_size * torch.randn(
            (self.sizes[0], self.rank), dtype=self.data_type
        )
        self.rel.weight.data = self.init_size * torch.randn(
            (self.sizes[1], 2 * self.rank), dtype=self.data_type
        )
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand(
            (self.sizes[1], self.rank), dtype=self.data_type
        ) - 1.0

        self.multi_c = args.multi_c
        if self.multi_c:
            c_init = torch.ones((self.sizes[1], 1), dtype=self.data_type)
        else:
            c_init = torch.ones((1, 1), dtype=self.data_type)
        self.c = nn.Parameter(c_init, requires_grad=True)

        # Register disease mask buffer
        mask = torch.zeros(self.sizes[0], dtype=torch.bool)
        if self.restrict_to_diseases and self.disease_ids is not None:
            mask[self.disease_ids] = True
            print("Number of disease nodes:", mask.sum().item())
        self.register_buffer("disease_mask", mask)

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores of queries against targets in embedding space."""
        # In AttH-like models, lhs_e is the tuple (final_query_vector, c)
        # In MuRP, lhs_e is the tuple (transformed_head, rel_vector, c)
        # rhs_e is always the untransformed tail embeddings (in tangent space)
        lhs, c = lhs_e
        return -hyp_distance_multi_c(lhs, rhs_e, c, eval_mode) ** 2


class RotH(BaseH):
    """Hyperbolic 2x2 Givens rotations"""

    def __init__(self, args, disease_ids=None, restrict_to_diseases=False):
        super(RotH, self).__init__(args, disease_ids=disease_ids, restrict_to_diseases=restrict_to_diseases)

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        if self.multi_c:
            c = F.softplus(self.c[queries[:, 1]])
        else:
            c = F.softplus(self.c.expand(queries.shape[0], -1))
            
        head = expmap0(self.entity(queries[:, 0]), c)
        rel1, rel2 = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel1 = expmap0(rel1, c)
        rel2 = expmap0(rel2, c)
        lhs = project(mobius_add(head, rel1, c), c)
        res1 = givens_rotations(self.rel_diag(queries[:, 1]), lhs)
        res2 = mobius_add(res1, rel2, c)
        # res2 is the final query vector, lhs_e in similarity_score
        return (res2, c), self.bh(queries[:, 0])


class RefH(BaseH):
    """Hyperbolic 2x2 Givens reflections"""

    def __init__(self, args, disease_ids=None, restrict_to_diseases=False):
        super(RefH, self).__init__(args, disease_ids=disease_ids, restrict_to_diseases=restrict_to_diseases)

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        if self.multi_c:
            c = F.softplus(self.c[queries[:, 1]])
        else:
            c = F.softplus(self.c.expand(queries.shape[0], -1))

        rel, _ = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel = expmap0(rel, c)
        lhs_e = givens_reflection(self.rel_diag(queries[:, 1]), self.entity(queries[:, 0]))
        lhs = expmap0(lhs_e, c)
        res = project(mobius_add(lhs, rel, c), c)
        # res is the final query vector, lhs_e in similarity_score
        return (res, c), self.bh(queries[:, 0])


class AttH(BaseH):
    """Hyperbolic attention model combining translations, reflections and rotations"""

    def __init__(self, args, disease_ids=None, restrict_to_diseases=False):
        super(AttH, self).__init__(args, disease_ids=disease_ids, restrict_to_diseases=restrict_to_diseases)
        self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.rel_diag.weight.data = 2 * torch.rand(
            (self.sizes[1], 2 * self.rank), dtype=self.data_type
        ) - 1.0

        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = self.init_size * torch.randn(
            (self.sizes[1], self.rank), dtype=self.data_type
        )

        self.act = nn.Softmax(dim=1)
        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda()
        else:
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        if self.multi_c:
            c = F.softplus(self.c[queries[:, 1]])
        else:
            c = F.softplus(self.c.expand(queries.shape[0], -1))

        head = self.entity(queries[:, 0])
        rot_mat, ref_mat = torch.chunk(self.rel_diag(queries[:, 1]), 2, dim=1)
        rot_q = givens_rotations(rot_mat, head).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head).view((-1, 1, self.rank))
        cands = torch.cat([ref_q, rot_q], dim=1)

        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)

        lhs = expmap0(att_q, c)
        rel, _ = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)
        # res is the final query vector, lhs_e in similarity_score
        return (res, c), self.bh(queries[:, 0])


class MurP(BaseH):
    """Hyperbolic translational model (MuRP) https://arxiv.org/pdf/1905.09791.pdf"""

    def __init__(self, args, disease_ids=None, restrict_to_diseases=False):
        super(MurP, self).__init__(args, disease_ids=disease_ids, restrict_to_diseases=restrict_to_diseases)
        
        # MuRP needs a relation vector (r_h) and a diagonal matrix (R)
        # We use self.rel_diag (rank d) for R
        # We override self.rel to be rank d (for r_h)
        
        self.rel = nn.Embedding(self.sizes[1], self.rank)
        self.rel.weight.data = self.init_size * torch.randn(
            (self.sizes[1], self.rank), dtype=self.data_type
        )
        
    def get_queries(self, queries):
        """Compute transformed head embedding h_s^{(r)} and retrieve r_h and c."""
        
        if self.multi_c: #we edit roiginal murp by allowing multiple curvatures
            
            c = F.softplus(self.c[queries[:, 1]])
        else:
            c = torch.ones(queries.shape[0], 1, dtype=self.data_type, device=self.c.device)

        # params in tangent space
        head_tangent = self.entity(queries[:, 0]) #head emb
        R_diag = self.rel_diag(queries[:, 1])  # Diagonal matrix R
        r_h_tangent = self.rel(queries[:, 1])  # Translation vector r_h
        
        # Compute transformed head: h_s^{(r)}
        lhs_tangent = R_diag * head_tangent
        lhs = expmap0(lhs_tangent, c) # map to hyperbolic space
        
        #  map r_h to hyperbolic space
        r_h = expmap0(r_h_tangent, c)
        
        return (lhs, r_h, c), self.bh(queries[:, 0])

    def similarity_score(self, lhs_e_tuple, rhs_e_tangent, eval_mode):
        """
        Compute score -d(h_s^{(r)}, h_o^{(r)})^2.
        lhs_e_tuple is (h_s^{(r)}, r_h, c) from get_queries.
        rhs_e_tangent is the untransformed tail embeddings from get_rhs.
        """
        
        # unpack the query-side info
        lhs, r_h, c = lhs_e_tuple  # lhs is h_s^{(r)} [B, D], r_h is [B, D], c is [B, 1]
        
        # transform tails (rhs) to get h_o^{(r)}
        if eval_mode:
            # rhs_e_tangent is [N, D] (all entities)
            B, N = lhs.shape[0], rhs_e_tangent.shape[0]

            # Broadcast c from [B, 1] to [B, N, 1]
            c_b = c.unsqueeze(1).expand(-1, N, -1)
            
            # Broadcast rhs_e_tangent from [N, D] to [B, N, D]
            rhs_e_tangent_b = rhs_e_tangent.unsqueeze(0).expand(B, N, -1)
            
            # shapes are now compatible: [B, N, D] and [B, N, 1]
            rhs = expmap0(rhs_e_tangent_b, c_b) # rhs is h_o (untransformed)

            # Broadcast r_h from [B, D] to [B, N, D]
            r_h_b = r_h.unsqueeze(1).expand(-1, N, -1)
            
            # Compute transformed rhs: h_o^{(r)} = h_o +_c r_h
            rhs_transformed = project(mobius_add(rhs, r_h_b, c_b), c_b)
            
            # Broadcast lhs (h_s^{(r)}) from [B, D] to [B, N, D]
            lhs_b = lhs.unsqueeze(1).expand(-1, N, -1)
            
            # Compute score using the edited elementwise function
            dist = hyp_distance_multi_c_elementwise(lhs_b, rhs_transformed, c_b)
            score = -(dist.squeeze(-1) ** 2) # Squeeze from [B, N, 1] to [B, N]
        else:
            # Batch-wise computation
            # rhs_e_tangent is [B, D], c is [B, 1]. Shapes are compatible.
            
            rhs = expmap0(rhs_e_tangent, c) # rhs is h_o (untransformed)
            
            # compute transformed rhs: h_o^{(r)} = h_o +_c r_h
            rhs_transformed = project(mobius_add(rhs, r_h, c), c)
            
            # compute score using the edited elementwise function
            dist = hyp_distance_multi_c_elementwise(lhs, rhs_transformed, c)
            score = -(dist ** 2) # Shape [B, 1] is correct
            
        return score
