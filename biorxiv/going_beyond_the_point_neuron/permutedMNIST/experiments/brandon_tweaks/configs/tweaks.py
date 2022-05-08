"""
Minor tweaks to existing numenta configs.
"""

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

from .base import (
    BASE,
    RESOURCES,
    LOCAL_DIR,
    make_config,
    as_search_config,
    MNIST_MEAN, MNIST_STD,
    CIFAR10_MEAN, CIFAR10_STD,
)

class BrandonSearchPrototypeExperiment(
    vernon_mixins.RezeroWeights,
    dendrites_mixins.PrototypeContext,
    cl_mixins.PermutedMNISTTaskIndices,
    b_experiments.BrandonDendriteContinualLearningExperiment
):
    pass


class BrandonSearchSIPrototypeExperiment(
    cl_mixins.SynapticIntelligence,
    vernon_mixins.RezeroWeights,
    dendrites_mixins.PrototypeContext,
    cl_mixins.PermutedMNISTTaskIndices,
    b_experiments.BrandonDendriteContinualLearningExperiment
):
    pass


class BrandonMLPExperiment(
    cl_mixins.PermutedMNISTTaskIndices,
    b_experiments.BrandonDendriteContinualLearningExperiment
):
    pass


SEARCH_PROTOTYPE_10 = as_search_config(
    PROTOTYPE_10,
    # hidden_sizes=(4096, 8192),
    # weight_sparsity=(0.0, 0.2, 0.5, 0.8),
    # num_layers=(3,),
    # lr = (5e-6, 1e-5, 5e-5),
    # kw_percent_on=(0.01, 0.05, 0.2, 0.5),
)
SEARCH_PROTOTYPE_10.update(
    experiment_class=BrandonSearchPrototypeExperiment,
    dataset_class=b_datasets.BrandonPermutedMNIST,
    tasks_to_validate=list(range(100)),
    **RESOURCES,
)
SEARCH_PROTOTYPE_10['train_model_args'] = dict(
    share_labels=True,
    num_labels=10)
SEARCH_PROTOTYPE_10["train_dataset_args"] = dict(
    transform=processing.get_dataset_transform('train', MNIST_MEAN, MNIST_STD),
    **SEARCH_PROTOTYPE_10["dataset_args"],
)
SEARCH_PROTOTYPE_10["valid_dataset_args"] = dict(
    transform=processing.get_dataset_transform('valid', MNIST_MEAN, MNIST_STD),
    **SEARCH_PROTOTYPE_10["dataset_args"],
)

SEARCH_PROTOTYPE_10_NOAUG = deepcopy(SEARCH_PROTOTYPE_10)
SEARCH_PROTOTYPE_10_NOAUG["train_dataset_args"].update(
    transform=processing.get_dataset_transform('valid', MNIST_MEAN, MNIST_STD))

# Search Prototype
# -----------------------------------------------
SEARCH_SI_PROTOTYPE_10 = as_search_config(SI_PROTOTYPE_10)
SEARCH_SI_PROTOTYPE_10.update(
    experiment_class=BrandonSearchSIPrototypeExperiment,
    dataset_class=b_datasets.BrandonPermutedMNIST,
    tasks_to_validate=list(range(100)),
    **RESOURCES,
)
SEARCH_SI_PROTOTYPE_10['si_args'].update(
    apply_to_dendrites=tune.grid_search([True, False])
)
SEARCH_SI_PROTOTYPE_10['train_model_args'] = dict(
    share_labels=True,
    num_labels=10)
SEARCH_SI_PROTOTYPE_10["train_dataset_args"] = dict(
    transform=processing.get_dataset_transform('train', MNIST_MEAN, MNIST_STD),
    **SEARCH_SI_PROTOTYPE_10["dataset_args"],
)
SEARCH_SI_PROTOTYPE_10["valid_dataset_args"] = dict(
    transform=processing.get_dataset_transform('valid', MNIST_MEAN, MNIST_STD),
    **SEARCH_SI_PROTOTYPE_10["dataset_args"],
)

SEARCH_SI_PROTOTYPE_10_NOAUG = deepcopy(SEARCH_SI_PROTOTYPE_10)
SEARCH_SI_PROTOTYPE_10_NOAUG["train_dataset_args"].update(
    transform=processing.get_dataset_transform('valid', MNIST_MEAN, MNIST_STD))

MLP_CONFIGS_WITH_RESOURCES = {
    f'brandon_{k}': {
        **v,
        **RESOURCES,
        'local_dir': LOCAL_DIR,
        'experiment_class': BrandonMLPExperiment,
        'dataset_class': b_datasets.BrandonPermutedMNIST,
        'tasks_to_validate': list(range(100)),
    }
    for k, v in deepcopy(MLP_CONFIGS).items()
}
for k, v in MLP_CONFIGS_WITH_RESOURCES.items():
    v["horizontal_flip_prob"] = 0.5
    v["train_dataset_args"] = dict(
        transform=processing.get_dataset_transform(
            'train', MNIST_MEAN, MNIST_STD,
            horizontal_flip_prob=v["horizontal_flip_prob"],
        ),
        **v["dataset_args"],
    )
    v["valid_dataset_args"] = dict(
        transform=processing.get_dataset_transform(
            'valid', MNIST_MEAN, MNIST_STD,
            horizontal_flip_prob=v["horizontal_flip_prob"],
        ),
        **v["dataset_args"],
    )

for k in deepcopy(list(MLP_CONFIGS_WITH_RESOURCES.keys())):
    MLP_CONFIGS_WITH_RESOURCES[f'{k}_noaug'] = deepcopy(MLP_CONFIGS_WITH_RESOURCES[k])
    MLP_CONFIGS_WITH_RESOURCES[f'{k}_noaug'].update(
        transform=processing.get_dataset_transform(
            'valid', MNIST_MEAN, MNIST_STD,
        )
    )

SEARCH_MLP_CONFIGS_WITH_RESOURCES = {
    f'search_{k}': v for k, v in deepcopy(MLP_CONFIGS_WITH_RESOURCES).items()}
for k in deepcopy(list(SEARCH_MLP_CONFIGS_WITH_RESOURCES.keys())):
    v = SEARCH_MLP_CONFIGS_WITH_RESOURCES[k]
    if "optimizer_args" not in v:
        v["optimizer_args"] = {}
    v["optimizer_args"].update(lr=tune.grid_search([1e-7, 1e-6, 1e-5]))
    v.update(
        samples=2,
        epochs=tune.grid_search([3, 5]),
    )


# ===================================================================
# Export configurations in this file
# ===================================================================
CONFIGS = dict(
    search_prototype_10=SEARCH_PROTOTYPE_10,
    search_prototype_10_noaug=SEARCH_PROTOTYPE_10_NOAUG,
    search_si_prototype_10=SEARCH_SI_PROTOTYPE_10,
    search_si_prototype_10_noaug=SEARCH_SI_PROTOTYPE_10_NOAUG,
    **MLP_CONFIGS_WITH_RESOURCES,
    **SEARCH_MLP_CONFIGS_WITH_RESOURCES,
)
