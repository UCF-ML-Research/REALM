import argparse

import torch
import torchvision.transforms as transforms


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def diff2clf(x):
    # [-1, 1] to [0, 1]
    return (x / 2) + 0.5


def clf2diff(x):
    # [0, 1] to [-1, 1]
    return (x - 0.5) * 2


def normalize(x):
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])(x)
