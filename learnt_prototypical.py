# https://github.com/VSainteuf/metric-guided-prototypes-pytorch/blob/master/torch_prototypes/modules/prototypical_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearntPrototypes(nn.Module):
    def __init__(self, model, n_prototypes, embedding_dim, prototypes=None,
                 squared=True, dist="euclidean", normalize=False, device="cuda"):
        """
        Args:
            model (nn.Module): feature extracting network
            n_prototypes (int): number of prototypes to use
            embedding_dim (int): dimension of the embedding space
            prototypes (tensor): Prototype tensor of shape (n_prototypes x embedding_dim),
            squared (bool): Whether to use the squared Euclidean distance or not
            dist (str): default 'euclidean', other possibility 'cosine'
            normalize (bool): l2 normalization of the features
            device (str): device on which to declare the prototypes (cpu/cuda)
        """
        super(LearntPrototypes, self).__init__()
        self.model = model
        self.prototypes = (nn.Parameter(torch.rand((n_prototypes, embedding_dim), device=device)).requires_grad_(True)
                           if prototypes is None else nn.Parameter(prototypes).requires_grad_(False))
        self.n_prototypes = n_prototypes
        self.squared = squared
        self.dist = dist
        self.normalize = normalize

    def forward(self, data):
        _, embeddings = self.model(data)
        if self.normalize:
            embeddings = F.normalize(embeddings, dim=1)

        b, c, h, w = embeddings.shape
        embeddings = embeddings.view(b, c, h * w).transpose(1, 2).contiguous().view(b * h * w, c)

        if self.dist == "cosine":
            dists = 1 - nn.CosineSimilarity(dim=-1)(embeddings[:, None, :], self.prototypes[None, :, :])
        else:
            dists = torch.norm(embeddings[:, None, :] - self.prototypes[None, :, :], dim=-1)
        if self.squared:
            dists = dists ** 2

        dists = dists.view(b, h * w, self.n_prototypes).transpose(1, 2).contiguous().view(b, self.n_prototypes, h, w)

        return -dists
