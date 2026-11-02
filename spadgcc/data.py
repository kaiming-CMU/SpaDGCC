from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch


def normalize_graph(graph, n):
    adjacency = sp.csr_matrix(graph, dtype=np.float32)
    if adjacency.shape != (n, n) or not np.isfinite(adjacency.data).all() or (adjacency.data < 0).any():
        raise ValueError("Adjacency must be a finite, nonnegative N x N matrix.")
    adjacency = adjacency + sp.eye(n, dtype=np.float32, format="csr")
    adjacency = sp.diags(1.0 / np.asarray(adjacency.sum(1)).ravel()) @ adjacency
    adjacency = adjacency.tocoo()
    return torch.sparse_coo_tensor(
        np.vstack((adjacency.row, adjacency.col)), adjacency.data,
        size=adjacency.shape, dtype=torch.float32,
        check_invariants=True,
    ).coalesce().to_dense()


def load_image_features(path, obs_names):
    with np.load(path, allow_pickle=False) as cache:
        names = np.asarray(cache["obs_names"], dtype=str)
        features = np.asarray(cache["features"], dtype=np.float32)
    if names.ndim != 1 or not len(names) or len(np.unique(names)) != len(names) or (names == "").any():
        raise ValueError("Image feature identifiers must be nonempty and unique.")
    if features.shape != (len(names), 1024) or not np.isfinite(features).all():
        raise ValueError("Image features must be a finite N x 1024 matrix.")
    index = {name: i for i, name in enumerate(names)}
    if any(name not in index for name in obs_names):
        raise ValueError("Image features are missing input spots.")
    return torch.from_numpy(features[[index[name] for name in obs_names]].copy())


@dataclass
class SpatialData:
    adata: ad.AnnData
    features: torch.Tensor
    sadj: torch.Tensor
    fadj: torch.Tensor
    image_features: torch.Tensor

    def to(self, device):
        return SpatialData(self.adata, *(value.to(device) for value in (
            self.features, self.sadj, self.fadj, self.image_features,
        )))


def load_data(path, image_features):
    adata = ad.read_h5ad(Path(path))
    features = np.asarray(adata.X.toarray() if sp.issparse(adata.X) else adata.X, dtype=np.float32)
    if features.ndim != 2 or min(features.shape) < 2 or not np.isfinite(features).all() or (features < 0).any():
        raise ValueError("Expression must contain at least two spots and genes with finite nonnegative values.")
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("Spot and gene identifiers must be unique.")
    graphs = []
    for key in ("sadj", "fadj"):
        if key not in adata.obsp:
            raise ValueError(f"Missing obsp['{key}'].")
        graphs.append(normalize_graph(adata.obsp[key], adata.n_obs))
    images = load_image_features(image_features, adata.obs_names)
    return SpatialData(adata, torch.from_numpy(features), *graphs, images)
