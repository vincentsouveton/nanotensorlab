import argparse
import os
import yaml
import torch

from nanotensorlab.data.datasets import build_dataset
from nanotensorlab.models.models import build_model
from nanotensorlab.training.trainer import Trainer


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default="outputs")

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
    # Create output directory
    # --------------------------------------------------

    exp_name = os.path.splitext(os.path.basename(args.config))[0]
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save experiment config for reproducibility
    torch.save(
        {"experiment_config": experiment_config},
        os.path.join(output_dir, "experiment_config.pt"),
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    trainer = Trainer(
        model=model,
        train_dataset=datasets["train"],
        val_dataset=datasets["val"],
        config=training_config,
        device=args.device,
        output_dir=output_dir,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
