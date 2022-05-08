import os
from copy import deepcopy

import numpy as np
import ray.tune as tune
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
from nupic.research.frameworks.continual_learning import mixins as cl_mixins
from nupic.research.frameworks.dendrites import DendriticMLP
from nupic.research.frameworks.dendrites import mixins as dendrites_mixins
from nupic.research.frameworks.vernon import mixins as vernon_mixins

from ...mlp import CONFIGS as MLP_CONFIGS
from ...prototype import PROTOTYPE_10
from ...si_prototype import SI_PROTOTYPE_10
from .. import (
    processing,
    trainables,
    experiments as b_experiments,
    datasets as b_datasets,
    mixins as b_mixins,
)

CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
MNIST_MEAN, MNIST_STD = (0.13062755,), (0.30810780,)
LOCAL_DIR = os.path.expanduser("~/nta/results/experiments/brandon_tweaks")

def seed_fn(spec):
    return np.random.randint(2, 10000)


def as_search_config(
        config,
        kw_percent_on=(0.05, 0.2),
        weight_sparsity=(0.0, 0.5, 0.8),
        hidden_sizes=(4096, 8192),
        num_layers=(3,),
        lr=(1e-5, 1e-4, 5e-4),
):
    search_hidden_sizes = []
    for hs in hidden_sizes:
        for nl in num_layers:
            search_hidden_sizes.append([hs] * nl)

    search_config = deepcopy(config)
    search_config["model_args"].update(
        kw_percent_on=tune.grid_search(list(kw_percent_on)),
        weight_sparsity=tune.grid_search(list(weight_sparsity)),
        hidden_sizes=tune.grid_search(list(search_hidden_sizes)))
    if "optimizer_args" not in search_config:
        search_config["optimizer_args"] = {}
    search_config["optimizer_args"].update(lr=tune.grid_search(list(lr)))
    return search_config


def make_config(
        base_config: Dict[str, Any],
        num_targets: int,
        num_tasks: int,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
):
    """Helper function to handle some common boilerplate/code duplication."""
    model_kwargs = model_kwargs or {}
    num_classes_per_task = num_targets // num_tasks
    config = deepcopy(base_config)
    config.update(
        num_tasks=num_tasks,
        num_classes=num_classes_per_task * num_tasks,
        tasks_to_validate=list(range(num_tasks)),
        **kwargs)
    config["model_args"].update(
        output_size=num_classes_per_task,  # Single output head shared by all tasks
        num_segments=num_tasks,
        **model_kwargs)
    config['train_model_args'].update(num_labels=num_classes_per_task)
    config["train_dataset_args"].update(num_tasks=num_tasks)
    config["valid_dataset_args"].update(num_tasks=num_tasks)
    return config

DATA_ROOT = os.path.expanduser("~/nta/results/data/")
DOWNLOAD = False  # Change to True if running for the first time

RESOURCES = dict(
    ray_trainable=trainables.BrandonRemoteProcessTrainable,
    workers=8,
    num_gpus=1,
    num_cpus=10,
    memory=20 * 1024 ** 3,
    # # https://docs.ray.io/en/releases-0.8.7/_modules/ray/tune/tune.html?highlight
    # =queue_trials#
    # resources_per_trial=dict(
    #     cpu=10,
    #     gpu=1,
    #     # I guess this is in units of Bytes, so multiply by 1024**3 to convert to GiB.
    #     # https://docs.ray.io/en/releases-0.8.7/_modules/ray/tune/resources.html
    #     ?highlight=extra_memory#
    #     memory=50 * 1024**3,
    # ),
    reuse_actors=True,
    fail_fast=True,
    queue_trials=False,
)

BASE = dict(
    # Results path
    local_dir=LOCAL_DIR,

    # Number of times to run (only makes sense if experiment is stochastic)
    num_samples=1,

    # Args forwarded to dataset constructors.
    train_dataset_args=dict(root=DATA_ROOT, download=DOWNLOAD),
    valid_dataset_args=dict(root=DATA_ROOT, download=DOWNLOAD),

    # Model spec.
    model_class=DendriticMLP,
    model_args=dict(
        input_size=784,
        output_size=None,  # Sub-configs need specify. Single output head shared by all tasks
        hidden_sizes=[2048, 2048],
        dim_context=784,
        kw=True,
        kw_percent_on=0.05,
        dendrite_weight_sparsity=0.0, # TODO: how dis work
        weight_sparsity=0.5, # TODO: how dis work
        context_percent_on=0.1, # TODO: how dis work
    ),
    train_model_args=dict(
        share_labels=True
    ),

    # NB: GLOBAL BATCH SIZE CAN'T BE ANY LARGER THAN MIN(EXAMPLES_PER_TASK)
    # Training spec.
    epochs=5,
    batch_size=512,
    val_batch_size=512,
    distributed=False,
    seed=tune.sample_from(seed_fn),
    loss_function=F.cross_entropy,
    optimizer_class=torch.optim.Adam,  # On permutedMNIST, Adam works better than

    # Resources.
    **RESOURCES,
)
