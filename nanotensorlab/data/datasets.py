import torch
from torch.utils.data import TensorDataset
from typing import Dict

from nanotensorlab.data.generators.gaussian import GaussianGenerator


# =====================================================
# List of available generators
# =====================================================

GENERATOR_REGISTRY = {
    "gaussian": GaussianGenerator,
    # "sine": SineGenerator,
    # "mixture": MixtureGaussianGenerator,
}


# =====================================================
# Dataset Builder
# =====================================================

def build_dataset(config: dict) -> Dict[str, TensorDataset]:
    """
    Build a dataset from config.
    """

    # -------------------------------------------------
    # 1. Seed
    # -------------------------------------------------
    if "seed" in config:
        torch.manual_seed(config["seed"])

    # -------------------------------------------------
    # 2. Select generator
    # -------------------------------------------------
    generator_name = config["generator"].lower()

    if generator_name not in GENERATOR_REGISTRY:
        raise ValueError(
            f"Unknown generator '{generator_name}'. "
            f"Available: {list(GENERATOR_REGISTRY.keys())}"
        )

    generator_cls = GENERATOR_REGISTRY[generator_name]
    generator = generator_cls(**config["params"])

    # -------------------------------------------------
    # 3. Sampling
    # -------------------------------------------------
    n_samples = config["sampling"]["n_samples"]
    data = generator.sample(n_samples)

    # -------------------------------------------------
    # 4. Split
    # -------------------------------------------------
    split_cfg = config["split"]

    n_train = int(split_cfg["train"] * n_samples)
    n_val = int(split_cfg["val"] * n_samples)
    n_test = n_samples - n_train - n_val

    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]

    return {
        "train": TensorDataset(train),
        "val": TensorDataset(val),
        "test": TensorDataset(test),
    }
