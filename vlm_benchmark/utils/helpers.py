"""General utility helpers for VLM benchmark."""

from typing import Dict, Any

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def import_class(class_path: str):
    """Dynamically import a class from module path."""
    module_path, class_name = class_path.rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def create_dataset(dataset_config: Dict[str, Any]):
    """Create dataset instance from config.

    Filters kwargs to only pass parameters the dataset constructor accepts,
    since different datasets have different signatures.
    """
    import inspect
    # Lazy import to avoid circular deps
    from vlm_benchmark.config import DATASET_REGISTRY

    dataset_type = dataset_config.get("type", "drivelm")
    dataset_class_path = DATASET_REGISTRY.get(dataset_type)

    if dataset_class_path is None:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    DatasetClass = import_class(dataset_class_path)

    dataset_kwargs = dataset_config.copy()
    dataset_kwargs.pop("type", None)

    # Filter to only kwargs accepted by the constructor
    sig = inspect.signature(DatasetClass.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    if not has_var_keyword:
        dataset_kwargs = {k: v for k, v in dataset_kwargs.items() if k in valid_params}

    return DatasetClass(**dataset_kwargs)
