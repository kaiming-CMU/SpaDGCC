import torch.nn as nn
import torch.nn.functional as F


IMAGE_TARGET_NORMALIZATION = {
    "method": "per_spot_layer_norm",
    "dimension": 1024,
    "eps": 1e-5,
    "elementwise_affine": False,
    "variance": "population",
}


def fixed_image_target(features):
    if features is None or features.ndim != 2 or features.shape[1] != 1024:
        raise ValueError("Image features must have shape N x 1024.")
    if not features.isfinite().all():
        raise ValueError("Image features must be finite.")
    return F.layer_norm(features.detach(), (1024,), eps=IMAGE_TARGET_NORMALIZATION["eps"])


class ReconstructionDecoder(nn.Module):
    def __init__(self, latent, hidden, output_dim):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(latent, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True)
        )
        self.output = nn.Linear(hidden, output_dim)

    def forward(self, fused):
        return self.output(self.hidden(fused))


def reconstruction_loss(prediction, target):
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("Reconstruction and target must share the same N x D shape.")
    return F.mse_loss(prediction, target.detach(), reduction="mean")
