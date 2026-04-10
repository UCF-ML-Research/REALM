from pathlib import Path

import yaml
import torch

from .guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from .utils import dict2namespace


def load_models(model_src, device):
    with open(Path(__file__).parent / 'diffusion_configs' / 'imagenet.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    config = dict2namespace(config)
    model_config = model_and_diffusion_defaults()
    model_config.update(vars(config.model))
    diffusion, _ = create_model_and_diffusion(**model_config)
    diffusion.load_state_dict(torch.load(model_src, map_location='cpu'))
    diffusion.eval().to(device)
    return diffusion
