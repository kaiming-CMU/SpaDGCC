from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch
from sklearn.cluster import KMeans


@dataclass
class BallSet:
    centers: torch.Tensor
    assignments: torch.Tensor
    sizes: torch.Tensor


def build_balls(embedding: torch.Tensor, p: int = 8) -> BallSet:
    if embedding.ndim != 2 or min(embedding.shape) < 1:
        raise ValueError("Each embedding must be a nonempty N x d matrix.")
    if not isinstance(p, int) or isinstance(p, bool) or p < 2:
        raise ValueError("p must be an integer >= 2.")
    if not torch.isfinite(embedding).all():
        raise ValueError("Embeddings must be finite.")
    data = embedding.detach().cpu().numpy()
    k = min(max(len(data) // p, 1), len(np.unique(data, axis=0)))
    if k == 1:
        labels = np.zeros(len(data), dtype=np.int64)
    else:
        labels = KMeans(n_clusters=k, n_init="auto").fit_predict(data)
    _, labels = np.unique(labels, return_inverse=True)
    ids = torch.as_tensor(labels, dtype=torch.long, device=embedding.device)
    k = int(labels.max()) + 1
    sizes = torch.bincount(ids, minlength=k).to(embedding.dtype)
    sums = embedding.new_zeros((k, embedding.shape[1])).index_add(0, ids, embedding)
    return BallSet(sums / sizes[:, None], ids, sizes)


def cross_view_relation(a: BallSet, b: BallSet, threshold: float = 0.1):
    if a.assignments.shape != b.assignments.shape:
        raise ValueError("Views must contain the same aligned samples.")
    ka, kb = len(a.centers), len(b.centers)
    counts = torch.bincount(
        a.assignments * kb + b.assignments, minlength=ka * kb
    ).reshape(ka, kb)
    denominator = torch.minimum(a.sizes[:, None], b.sizes[None, :])
    mask = (counts / denominator > threshold).to(a.centers.dtype)
    dice = 2 * counts / (a.sizes[:, None] + b.sizes[None, :])
    return mask, dice


def pair_loss(a: BallSet, b: BallSet, temperature: float = 1., threshold: float = .1):
    ka, kb = len(a.centers), len(b.centers)
    cross, dice = cross_view_relation(a, b, threshold)
    eye_a = torch.eye(ka, device=cross.device, dtype=cross.dtype)
    eye_b = torch.eye(kb, device=cross.device, dtype=cross.dtype)
    mask = torch.cat((torch.cat((eye_a, cross), 1),
                      torch.cat((cross.T, eye_b), 1)), 0)
    cross_weights = cross * dice
    weights = torch.cat((torch.cat((eye_a, cross_weights), 1),
                         torch.cat((cross_weights.T, eye_b), 1)), 0)
    centers = torch.cat((a.centers, b.centers))
    norms = centers.norm(p=2, dim=1, keepdim=True)
    similarity = centers @ centers.T / (norms @ norms.T + 1e-12)
    positive_logits = mask * similarity / temperature
    negative_logits = (1 - mask) * similarity / temperature
    log_negative_sum = torch.logsumexp(negative_logits, dim=1, keepdim=True)
    log_denominator = torch.logaddexp(positive_logits, log_negative_sum)
    return (weights * (log_denominator - positive_logits)).sum() / weights.sum()


class MultiviewGCLoss(torch.nn.Module):
    def __init__(self, temperature=1., p=8, threshold=.1):
        super().__init__()
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive.")
        if not np.isfinite(threshold) or not 0 <= threshold < 1:
            raise ValueError("threshold must be in [0, 1).")
        if not isinstance(p, int) or isinstance(p, bool) or p < 2:
            raise ValueError("p must be an integer >= 2.")
        self.t, self.p, self.threshold = temperature, p, threshold
        self.last_num_balls = []

    def forward(self, embeddings):
        if len(embeddings) < 2:
            raise ValueError("At least two views are required.")
        first = embeddings[0]
        for embedding in embeddings:
            if (embedding.shape != first.shape or embedding.device != first.device
                    or embedding.dtype != first.dtype):
                raise ValueError("Views need equal N x d shapes, devices and dtypes.")
        balls = [build_balls(embedding, self.p) for embedding in embeddings]
        self.last_num_balls = [len(ball.centers) for ball in balls]
        losses = [pair_loss(a, b, self.t, self.threshold) for a, b in combinations(balls, 2)]
        return torch.stack(losses).mean()
