"""Euclidean Knowledge Graph embedding models where embeddings are in complex space,
   with optional disease‐node restriction."""

import torch
from torch import nn

from models.base import KGModel

COMPLEX_MODELS = ["ComplEx", "RotatE"]


class BaseC(KGModel):
    """Complex Knowledge Graph Embedding models.

    Adds:
        disease_ids: list or tensor of entity‐IDs corresponding to diseases
        restrict_to_diseases: if True, only those IDs will be “active”
        disease_mask: a buffer bool mask over entities

    Attributes:
        embeddings: complex embeddings for entities and relations
    """

    def __init__(self, args, disease_ids=None, restrict_to_diseases=False):
        """Initialize a Complex KGModel, optionally restricting to disease nodes."""
        super(BaseC, self).__init__(
            args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias, args.init_size
        )
        assert self.rank % 2 == 0, "Complex models require even embedding dimension"
        # halve the rank for real/imag components
        self.rank = self.rank // 2

        # core embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * self.rank, sparse=True)
            for s in self.sizes[:2]
        ])
        # init them
        self.embeddings[0].weight.data = self.init_size * self.embeddings[0].weight.to(self.data_type)
        self.embeddings[1].weight.data = self.init_size * self.embeddings[1].weight.to(self.data_type)

        # disease filtering setup (only used if restrict_to_diseases=True)
        self.disease_ids = disease_ids
        self.restrict_to_diseases = restrict_to_diseases

        # build and register a boolean mask over all entities
        mask = torch.zeros(self.sizes[0], dtype=torch.bool)
        if self.restrict_to_diseases and self.disease_ids is not None:
            mask[self.disease_ids] = True
            print("Number of disease nodes (Complex):", mask.sum().item())
        self.register_buffer("disease_mask", mask)

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            # at eval time we could use self.disease_mask to filter downstream
            return self.embeddings[0].weight, self.bt.weight
        else:
            return self.embeddings[0](queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores in complex space."""
        # split into real & imag parts
        lhs_re, lhs_im = lhs_e[:, :self.rank], lhs_e[:, self.rank:]
        rhs_re, rhs_im = rhs_e[:, :self.rank], rhs_e[:, self.rank:]
        if eval_mode:
            return lhs_re @ rhs_re.transpose(0, 1) + lhs_im @ rhs_im.transpose(0, 1)
        else:
            return torch.sum(lhs_re * rhs_re + lhs_im * rhs_im, 1, keepdim=True)

    def get_complex_embeddings(self, queries):
        """Get (re,im) tuples for head, rel, tail."""
        head = self.embeddings[0](queries[:, 0])
        rel  = self.embeddings[1](queries[:, 1])
        tail = self.embeddings[0](queries[:, 2])
        head = head[:, :self.rank], head[:, self.rank:]
        rel  = rel[:, :self.rank],  rel[:, self.rank:]
        tail = tail[:, :self.rank], tail[:, self.rank:]
        return head, rel, tail

    def get_factors(self, queries):
        """Compute norms for regularization."""
        head, rel, tail = self.get_complex_embeddings(queries)
        head_f = torch.sqrt(head[0]**2 + head[1]**2)
        rel_f  = torch.sqrt(rel[0]**2  + rel[1]**2)
        tail_f = torch.sqrt(tail[0]**2 + tail[1]**2)
        return head_f, rel_f, tail_f


class ComplEx(BaseC):
    """Simple complex model http://proceedings.mlr.press/v48/trouillon16.pdf"""

    def get_queries(self, queries):
        head_e, rel_e, _ = self.get_complex_embeddings(queries)
        # (a+bi)(c+di) = (ac−bd) + i(ad+bc)
        lhs_re = head_e[0] * rel_e[0] - head_e[1] * rel_e[1]
        lhs_im = head_e[0] * rel_e[1] + head_e[1] * rel_e[0]
        lhs_e  = torch.cat([lhs_re, lhs_im], dim=1)
        return lhs_e, self.bh(queries[:, 0])


class RotatE(BaseC):
    """Rotations in complex space https://openreview.net/pdf?id=HkgEQnRqYQ"""

    def get_queries(self, queries):
        head_e, rel_e, _ = self.get_complex_embeddings(queries)
        # normalize the relation to unit‐circle
        norm = torch.sqrt(rel_e[0] ** 2 + rel_e[1] ** 2)
        cos  = rel_e[0] / norm
        sin  = rel_e[1] / norm
        # rotate: (a+bi) * (cos + i sin)
        lhs_re = head_e[0] * cos - head_e[1] * sin
        lhs_im = head_e[0] * sin + head_e[1] * cos
        lhs_e  = torch.cat([lhs_re, lhs_im], dim=1)
        return lhs_e, self.bh(queries[:, 0])
