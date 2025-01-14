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

from .. import (
    processing,
    trainables,
    experiments as b_experiments,
    datasets as b_datasets,
    mixins as b_mixins,
)

from .base import (
    BASE,
    make_config,
    as_search_config,
    MNIST_MEAN, MNIST_STD,
    CIFAR10_MEAN, CIFAR10_STD,
)


# Reminder: MRO will use the first defined method encountered in the class list:
class SplitDataExperiment(
    vernon_mixins.RezeroWeights,
    b_mixins.BrandonPrototypeContext,
    b_mixins.SplitDatasetTaskIndices,
    b_experiments.BrandonDendriteContinualLearningExperiment
):
    pass


class BrandonSearchSISplitDataExperiment(
    cl_mixins.SynapticIntelligence,
    vernon_mixins.RezeroWeights,
    b_mixins.BrandonPrototypeContext,
    b_mixins.SplitDatasetTaskIndices,
    b_experiments.BrandonDendriteContinualLearningExperiment
):
    pass


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
# SEARCH_SPLIT_MNIST = as_search_config(SPLIT_MNIST)
SEARCH_SPLIT_MNIST = as_search_config(
    SPLIT_MNIST,
    kw_percent_on=(0.05, 0.2),
    weight_sparsity=(0.0, 0.5, 0.8),
    hidden_sizes=(256, 512, 4096),
    num_layers=(2, 3),
    lr=(1e-5, 1e-4, 5e-4, 1e-3))
SEARCH_SPLIT_MNIST_NOAUG = deepcopy(SEARCH_SPLIT_MNIST)
SEARCH_SPLIT_MNIST_NOAUG["train_dataset_args"].update(
    transform=processing.get_dataset_transform('valid', MNIST_MEAN, MNIST_STD))

SEARCH_SPLIT_CIFAR10 = as_search_config(
    SPLIT_CIFAR10,
    num_layers=(3,),
    hidden_sizes=(2048, 4096),
    lr=(1e-5, 1e-4, 1e-3),
)
SEARCH_SPLIT_CIFAR10_NOAUG = as_search_config(SPLIT_CIFAR10)
SEARCH_SPLIT_CIFAR10_NOAUG["train_dataset_args"].update(
    transform=processing.get_dataset_transform('valid', CIFAR10_MEAN, CIFAR10_STD))

SEARCH_CIFAR100_2 = as_search_config(SPLIT_CIFAR100_2)
SEARCH_CIFAR100_5 = as_search_config(SPLIT_CIFAR100_5)

SEARCH_CIFAR100_10 = as_search_config(
    SPLIT_CIFAR100_10,
    kw_percent_on=(0.01, 0.05, 0.2),
    weight_sparsity=(0.0, 0.5, 0.8),
    hidden_sizes=(4096, 8192),
    num_layers=(2, 3),
    lr=(5e-5, 1e-4, 5e-4))
SEARCH_CIFAR100_10_NOAUG = deepcopy(SEARCH_CIFAR100_10)
SEARCH_CIFAR100_10_NOAUG["train_dataset_args"].update(
    transform=processing.get_dataset_transform('valid', CIFAR10_MEAN, CIFAR10_STD))


SEARCH_SI_SPLIT_CIFAR10 = deepcopy(SEARCH_SPLIT_CIFAR10)
SEARCH_SI_SPLIT_CIFAR10['si_args'] = dict(
    c=0.1,
    damping=0.1,
    apply_to_dendrites=tune.grid_search([True, False])
)
# The SI paper reports not resetting the Adam
# optimizer between tasks, and this
# works well with dendrites too
# experiment_class = BrandonSearchSIPrototypeExperiment,
SEARCH_SI_SPLIT_CIFAR10.update(
    experiment_class=BrandonSearchSISplitDataExperiment,
    reset_optimizer_after_tasks=False,
    epochs=tune.grid_search([5, 10]),
)

SEARCH_SI_SPLIT_CIFAR10_NOAUG = deepcopy(SEARCH_SI_SPLIT_CIFAR10)
SEARCH_SI_SPLIT_CIFAR10_NOAUG["train_dataset_args"].update(
    transform=processing.get_dataset_transform('valid', CIFAR10_MEAN, CIFAR10_STD))

SEARCH_SI_SPLIT_MNIST = deepcopy(SEARCH_SPLIT_MNIST)
SEARCH_SI_SPLIT_MNIST['si_args'] = dict(
    c=0.1,
    damping=0.1,
    apply_to_dendrites=tune.grid_search([True, False])
)
# The SI paper reports not resetting the Adam
# optimizer between tasks, and this
# works well with dendrites too
# experiment_class = BrandonSearchSIPrototypeExperiment,
SEARCH_SI_SPLIT_MNIST.update(
    experiment_class=BrandonSearchSISplitDataExperiment,
    reset_optimizer_after_tasks=False,
    epochs=tune.grid_search([5, 10]),
)

SEARCH_SI_SPLIT_MNIST_NOAUG = deepcopy(SEARCH_SI_SPLIT_MNIST)
SEARCH_SI_SPLIT_MNIST_NOAUG["train_dataset_args"].update(
    transform=processing.get_dataset_transform('valid', MNIST_MEAN, MNIST_STD))

# Export configurations in this file
CONFIGS = dict(
    split_mnist=SPLIT_MNIST,
    split_cifar10=SPLIT_CIFAR10,
    split_cifar100_2=SPLIT_CIFAR100_2,
    split_cifar100_5=SPLIT_CIFAR100_5,
    split_cifar100_10=SPLIT_CIFAR100_10,
    # Hyperparameter search configs:
    search_split_mnist=SEARCH_SPLIT_MNIST,
    search_split_mnist_noaug=SEARCH_SPLIT_MNIST_NOAUG,
    search_si_split_mnist=SEARCH_SI_SPLIT_MNIST,
    search_si_split_mnist_noaug=SEARCH_SI_SPLIT_MNIST,
    search_cifar10=SEARCH_SPLIT_CIFAR10,
    search_cifar10_noaug=SEARCH_SPLIT_CIFAR10_NOAUG,
    search_cifar100_2=SEARCH_CIFAR100_2,
    search_cifar100_5=SEARCH_CIFAR100_5,
    search_cifar100_10=SEARCH_CIFAR100_10,
    search_cifar100_10_noaug=SEARCH_CIFAR100_10_NOAUG,
    search_si_split_cifar10=SEARCH_SI_SPLIT_CIFAR10,
    search_si_split_cifar10_noaug=SEARCH_SI_SPLIT_CIFAR10_NOAUG,
)
