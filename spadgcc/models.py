import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GraphConvolution
from .reconstruction import ReconstructionDecoder


class GatedImageResidual(nn.Module):
    def __init__(self, latent, gate_init=0.1):
        super().__init__()
        if latent < 2:
            raise ValueError("latent must be at least 2.")
        if not math.isfinite(gate_init) or not 0 < gate_init < 1:
            raise ValueError("gate_init must be strictly between 0 and 1.")
        self.input_norm = nn.LayerNorm(1024, elementwise_affine=False, eps=1e-5)
        self.image_adapter = nn.Linear(1024, latent)
        self.output_norm = nn.LayerNorm(latent, elementwise_affine=False, eps=1e-5)
        self.gene_norm = nn.LayerNorm(latent, elementwise_affine=False, eps=1e-5)
        self.gate = nn.Linear(2 * latent, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(gate_init / (1 - gate_init)))

    def forward(self, gene, image):
        if image is None or image.shape != (gene.shape[0], 1024):
            raise ValueError("Image features must have shape N x 1024.")
        if image.device != gene.device or image.dtype != gene.dtype:
            raise ValueError("Image and gene features must share device and dtype.")
        residual = self.output_norm(self.image_adapter(self.input_norm(image.detach())))
        gate = torch.sigmoid(self.gate(torch.cat((self.gene_norm(gene), residual), dim=1)))
        return gene + gate * residual, residual, gate


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return self.gc2(x, adj)


class MLP_L(nn.Module):
    def __init__(self, n_mlp):
        super().__init__()
        self.wl = nn.Linear(n_mlp, n_mlp)

    def forward(self, mlp_in):
        return self.wl(mlp_in)


class SpaDGCC(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout, gate_init=0.1):
        super().__init__()
        self.SGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.FGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.dropout = dropout
        self.meta = nn.Parameter(torch.Tensor([0.1]))
        self.meta.data.clamp_(0, 1)
        self.MLP = nn.Sequential(nn.Linear(3 * nhid2, nhid2))
        self.MLP_L = MLP_L(nhid2)
        self.image_fusion = GatedImageResidual(nhid2, gate_init)
        self.decoder = ReconstructionDecoder(nhid2, nhid1, nfeat)
        self.image_decoder = ReconstructionDecoder(nhid2, nhid1, 1024)

    def forward(self, x, sadj, fadj, image_features=None, return_details=False):
        spatial = self.SGCN(x, sadj)
        feature = self.FGCN(x, fadj)
        conadj = self.meta * fadj + (1 - self.meta) * sadj
        complementary = self.CGCN(x, conadj)
        stacked = torch.stack([spatial, complementary, feature], dim=1)
        attention = F.normalize(self.MLP_L(stacked), p=2)
        combined = torch.cat((attention[:, 0].mul(spatial),
                              attention[:, 1].mul(complementary),
                              attention[:, 2].mul(feature)), dim=1)
        gene = self.MLP(combined)
        fused, residual, gate = self.image_fusion(gene, image_features)
        outputs = (fused, spatial, feature, complementary)
        if not return_details:
            return outputs
        details = {
            "gene_fused": gene,
            "image_residual": residual,
            "image_gate": gate,
            "reconstructed": self.decoder(fused),
            "image_reconstructed": self.image_decoder(fused),
        }
        return outputs, details
