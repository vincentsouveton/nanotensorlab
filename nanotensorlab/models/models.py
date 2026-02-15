from typing import Dict

from nanotensorlab.models.samplers.affine_flow import AffineFlow


MODEL_REGISTRY = {
    "affine_flow": AffineFlow
}


def build_model(config: dict):
    """
    Build a model from config.
    """

    model_name = config["model"].lower()

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[model_name]
    return model_cls(**config["params"])
