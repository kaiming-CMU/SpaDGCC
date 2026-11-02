import argparse
import json
import os
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from spadgcc.data import load_data
from spadgcc.granular import MultiviewGCLoss
from spadgcc.metrics import clustering_metrics
from spadgcc.models import SpaDGCC
from spadgcc.reconstruction import fixed_image_target, reconstruction_loss


def train(config, base_dir):
    for name in ("epochs", "hidden", "latent", "clusters"):
        if not isinstance(config[name], int) or isinstance(config[name], bool) or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not np.isfinite(config["lr"]) or config["lr"] <= 0 or not 0 <= config["dropout"] < 1:
        raise ValueError("Require positive finite lr and dropout in [0, 1).")
    requested = config["device"]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    output = base_dir / config["output"]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory must be empty.")
    data = load_data(base_dir / config["h5ad"], base_dir / config["image_features"]).to(device)
    if config["clusters"] > data.adata.n_obs:
        raise ValueError("clusters cannot exceed the number of spots.")
    label_key = config["label_key"]
    if label_key not in data.adata.obs or not data.adata.obs[label_key].notna().any():
        raise ValueError("Evaluation requires an annotation column with labeled spots.")
    model = SpaDGCC(data.features.shape[1], config["hidden"], config["latent"],
                    config["dropout"], gate_init=config["gate_init"]).to(device)
    image_target = fixed_image_target(data.image_features)
    criterion = MultiviewGCLoss(config["temperature"], config["p"], config["threshold"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=0.0)
    history = []
    with threadpool_limits(limits=1):
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            outputs, details = model(data.features, data.sadj, data.fadj,
                                     data.image_features, return_details=True)
            fused, spatial, feature, complementary = outputs
            contrast = criterion([spatial, feature, complementary, fused])
            gene = reconstruction_loss(details["reconstructed"], data.features)
            image = reconstruction_loss(details["image_reconstructed"], image_target)
            loss = gene + image + contrast
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}.")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"Missing or nonfinite gradient: {name}.")
            optimizer.step()
            row = {"epoch": epoch, "loss_total": float(loss.detach()),
                   "loss_con": float(contrast.detach()), "loss_gene": float(gene.detach()),
                   "loss_image": float(image.detach())}
            history.append(row)
            if epoch == 1 or epoch % 10 == 0 or epoch == config["epochs"]:
                print(f"epoch={epoch}/{config['epochs']} loss={row['loss_total']:.6f}", flush=True)
        model.eval()
        with torch.no_grad():
            outputs = model(data.features, data.sadj, data.fadj, data.image_features)
            embeddings = tuple(value.cpu().numpy() for value in outputs)
        if not all(np.isfinite(value).all() for value in embeddings):
            raise FloatingPointError("Nonfinite final embeddings.")
        predictions = KMeans(n_clusters=config["clusters"], n_init=20).fit_predict(embeddings[0])
    scores = clustering_metrics(data.adata.obs[label_key].to_numpy(), predictions)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    pd.DataFrame({"barcode": data.adata.obs_names, "cluster": predictions,
                  "ground_truth": data.adata.obs[label_key].to_numpy()}).to_csv(output / "clusters.csv", index=False)
    np.savez_compressed(output / "embedding.npz",
                        **dict(zip(("embedding", "spatial", "feature", "complementary"), embeddings)),
                        obs_names=np.asarray(data.adata.obs_names, dtype=str))
    torch.save({"model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "config": dict(config), "n_features": data.features.shape[1],
                "var_names": list(data.adata.var_names)}, output / "model.pt")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(scores, indent=2, allow_nan=False) + "\n")
    pd.DataFrame([scores]).to_csv(output / "metrics.csv", index=False)
    print(json.dumps(scores, indent=2), flush=True)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if args.device is not None:
        config["device"] = args.device
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.output is not None:
        config["output"] = str(args.output.resolve())
    train(config, config_path.parent)


if __name__ == "__main__":
    main()
