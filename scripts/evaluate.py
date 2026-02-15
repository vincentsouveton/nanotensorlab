# scripts/evaluate.py

import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from data.datasets import build_dataset
from models.models import build_model


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def move_batch_to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return tuple(b.to(device) for b in batch)
    return batch.to(device)


@torch.no_grad()
def evaluate(model, dataset, batch_size, device):

    model.eval()
    model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    total_loss = 0.0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        loss = model.loss(batch)
        total_loss += loss.item()

    return total_loss / len(loader)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    # --------------------------------------------------
    # Load experiment config
    # --------------------------------------------------

    experiment_config = load_yaml(args.config)

    data_config = load_yaml(experiment_config["data"])
    model_config = load_yaml(experiment_config["model"])
    training_config = load_yaml(experiment_config["training"])

    # --------------------------------------------------
    # Build dataset & model
    # --------------------------------------------------

    datasets = build_dataset(data_config)
    model = build_model(model_config)

    # --------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    val_loss = evaluate(
        model=model,
        dataset=datasets["val"],
        batch_size=training_config["batch_size"],
        device=args.device,
    )

    print(f"Validation loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
