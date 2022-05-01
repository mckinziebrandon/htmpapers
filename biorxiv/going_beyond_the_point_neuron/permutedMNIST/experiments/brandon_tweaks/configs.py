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

from ..prototype import PROTOTYPE_10
from . import (
    processing,
    experiments as b_experiments,
    datasets as b_datasets,
    mixins as b_mixins,
)


def as_search_config(
        config,
        kw_percent_on=(0.01, 0.05, 0.2),
        weight_sparsity=(0.5, 0.8, 0.95),
        hidden_sizes=(2048, 4096, 8192),
        num_layers=(2, 3),
        lr=(1e-4, 5e-4, 5e-3),
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


# Reminder: MRO will use the first defined method encountered in the class list:
class SplitDataExperiment(
    vernon_mixins.RezeroWeights,
    b_mixins.BrandonPrototypeContext,
    b_mixins.SplitDatasetTaskIndices,
    b_experiments.BrandonDendriteContinualLearningExperiment
):
    pass


class BrandonSearchPrototypeExperiment(
    vernon_mixins.RezeroWeights,
    dendrites_mixins.PrototypeContext,
    cl_mixins.PermutedMNISTTaskIndices,
    b_experiments.BrandonDendriteContinualLearningExperiment
):
    pass


def seed_fn(spec):
    return np.random.randint(2, 10000)


DATA_ROOT = os.path.expanduser("~/nta/results/data/")
DOWNLOAD = False  # Change to True if running for the first time
BASE = dict(
    # Results path
    local_dir=os.path.expanduser("~/nta/results/experiments/dendrites"),

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
                                       # SGD with default hyperparameter settings
)

# -----------------------------------------------
# Split MNIST
# -----------------------------------------------
SPLIT_MNIST = make_config(
    base_config=BASE,
    num_targets=10,
    num_tasks=5,
    model_kwargs=dict(
        kw_percent_on=0.05,
        weight_sparsity=0.8,
        hidden_sizes=[8192, 8192]),
    experiment_class=SplitDataExperiment,
    dataset_class=b_datasets.SplitMNIST,
    optimizer_args=dict(lr=1e-4))
MNIST_MEAN, MNIST_STD = (0.13062755,), (0.30810780,)
SPLIT_MNIST["train_dataset_args"].update(
    transform=processing.get_dataset_transform('train', MNIST_MEAN, MNIST_STD))
SPLIT_MNIST["valid_dataset_args"].update(
    transform=processing.get_dataset_transform('valid', MNIST_MEAN, MNIST_STD))

# -----------------------------------------------
# Split CIFAR-10
# -----------------------------------------------
SPLIT_CIFAR10 = make_config(
    base_config=BASE,
    num_targets=10,
    num_tasks=5,
    model_kwargs=dict(
        input_size=32 * 32 * 3,
        dim_context=32 * 32 * 3,
        kw_percent_on=0.05,
        weight_sparsity=0.8,
        hidden_sizes=[8192, 8192]),
    experiment_class=SplitDataExperiment,
    dataset_class=b_datasets.SplitCIFAR10,
    optimizer_args=dict(lr=1e-4))
CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
SPLIT_CIFAR10["train_dataset_args"].update(
    transform=processing.get_dataset_transform('train', CIFAR10_MEAN, CIFAR10_STD))
SPLIT_CIFAR10["valid_dataset_args"].update(
    transform=processing.get_dataset_transform('valid', CIFAR10_MEAN, CIFAR10_STD))

# -----------------------------------------------
# Split CIFAR-100
# -----------------------------------------------
SPLIT_CIFAR100_BASE = deepcopy(BASE)
SPLIT_CIFAR100_BASE.update(
    experiment_class=SplitDataExperiment,
    dataset_class=b_datasets.SplitCIFAR100,
    optimizer_args=dict(lr=1e-4))
SPLIT_CIFAR100_BASE["model_args"].update(
    input_size=32 * 32 * 3,
    dim_context=32 * 32 * 3,
    kw_percent_on=0.05,
    weight_sparsity=0.8,
    hidden_sizes=[8192, 8192])
SPLIT_CIFAR100_BASE["train_dataset_args"].update(
    transform=processing.get_dataset_transform('train', CIFAR10_MEAN, CIFAR10_STD))
SPLIT_CIFAR100_BASE["valid_dataset_args"].update(
    transform=processing.get_dataset_transform('valid', CIFAR10_MEAN, CIFAR10_STD))

SPLIT_CIFAR100_2 = make_config(
    base_config=SPLIT_CIFAR100_BASE,
    num_targets=100,
    num_tasks=2)
SPLIT_CIFAR100_5 = make_config(
    base_config=SPLIT_CIFAR100_BASE,
    num_targets=100,
    num_tasks=5)
SPLIT_CIFAR100_10 = make_config(
    base_config=SPLIT_CIFAR100_BASE,
    num_targets=100,
    num_tasks=10)

# -----------------------------------------------
# Hyperparameter Search Configs
# -----------------------------------------------
SEARCH_SPLIT_MNIST = as_search_config(SPLIT_MNIST)
SEARCH_SPLIT_CIFAR10 = as_search_config(SPLIT_CIFAR10)
SEARCH_SPLIT_CIFAR10_NOAUG = as_search_config(SPLIT_CIFAR10)
SEARCH_SPLIT_CIFAR10_NOAUG["train_dataset_args"].update(
    transform=processing.get_dataset_transform('valid', CIFAR10_MEAN, CIFAR10_STD))
SEARCH_CIFAR100_2 = as_search_config(SPLIT_CIFAR100_2)
SEARCH_CIFAR100_5 = as_search_config(SPLIT_CIFAR100_5)
SEARCH_CIFAR100_10 = as_search_config(SPLIT_CIFAR100_10)

SEARCH_PROTOTYPE_10 = as_search_config(PROTOTYPE_10)
SEARCH_PROTOTYPE_10.update(
    experiment_class=BrandonSearchPrototypeExperiment,
    tasks_to_validate=list(range(100)))
SEARCH_PROTOTYPE_10['train_model_args'] = dict(
    share_labels=True,
    num_labels=10)

# Export configurations in this file
CONFIGS = dict(
    split_mnist=SPLIT_MNIST,
    split_cifar10=SPLIT_CIFAR10,
    split_cifar100_2=SPLIT_CIFAR100_2,
    split_cifar100_5=SPLIT_CIFAR100_5,
    split_cifar100_10=SPLIT_CIFAR100_10,
    # Hyperparameter search configs:
    search_split_mnist=SEARCH_SPLIT_MNIST,
    search_cifar10=SEARCH_SPLIT_CIFAR10,
    search_cifar10_no_aug=SEARCH_SPLIT_CIFAR10_NOAUG,
    search_cifar100_2=SEARCH_CIFAR100_2,
    search_cifar100_5=SEARCH_CIFAR100_5,
    search_cifar100_10=SEARCH_CIFAR100_10,
    search_prototype_10=SEARCH_PROTOTYPE_10,
)
