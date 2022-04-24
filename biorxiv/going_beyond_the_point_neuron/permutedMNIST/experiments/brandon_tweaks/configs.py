
import os
from copy import deepcopy

import numpy as np
import ray.tune as tune
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from nupic.research.frameworks.continual_learning import mixins as cl_mixins
from nupic.research.frameworks.dendrites import DendriticMLP
from nupic.research.frameworks.dendrites import mixins as dendrites_mixins
from nupic.research.frameworks.dendrites.dendrite_cl_experiment import (
    DendriteContinualLearningExperiment,
)
from nupic.research.frameworks.pytorch.datasets import PermutedMNIST
from nupic.research.frameworks.vernon import mixins as vernon_mixins
from nupic.torch.modules import KWinners

from . import (
    experiments as b_experiments,
    datasets as b_datasets,
    mixins as b_mixins,
)


# Reminder: MRO will use the first defined method encountered in the class list:
class SplitDataExperiment(vernon_mixins.RezeroWeights,
                           b_experiments.BrandonPrototypeContext,
                           b_mixins.SplitDatasetTaskIndices,
                           b_experiments.BrandonDendriteContinualLearningExperiment):
    pass


def seed_fn(spec):
    return np.random.randint(2, 10000)


BASE = dict(
    dataset_args=dict(
        root=os.path.expanduser("~/nta/results/data/"),
        download=True,  # Change to True if running for the first time
    ),

    num_samples=1,

    # Results path
    local_dir=os.path.expanduser("~/nta/results/experiments/dendrites"),

    model_class=DendriticMLP,
    model_args=dict(
        input_size=784,
        output_size=None,  # Sub-configs need specify. Single output head shared by all tasks
        hidden_sizes=[2048, 2048],
        dim_context=784,
        kw=True,
        kw_percent_on=0.05,
        dendrite_weight_sparsity=0.0,
        weight_sparsity=0.5,
        context_percent_on=0.1,
    ),

    batch_size=256,
    val_batch_size=512,
    tasks_to_validate=[1, 4, 9, 24, 49, 99],
    distributed=False,
    seed=tune.sample_from(seed_fn),

    loss_function=F.cross_entropy,
    optimizer_class=torch.optim.Adam,  # On permutedMNIST, Adam works better than
                                       # SGD with default hyperparameter settings
)

# NB: GLOBAL BATCH SIZE CAN'T BE ANY LARGER THAN MIN(EXAMPLES_PER_TASK)

NUM_TASKS = 5
NUM_CLASSES_PER_TASK = 2
BATCH_SIZE = 512

SPLIT_MNIST = deepcopy(BASE)
SPLIT_MNIST.update(
    experiment_class=SplitDataExperiment,
    dataset_class=b_datasets.SplitMNIST,
    batch_size=BATCH_SIZE,
    val_batch_size=BATCH_SIZE,
    num_tasks=NUM_TASKS,
    num_classes=NUM_CLASSES_PER_TASK * NUM_TASKS,
    epochs=5,
    optimizer_args=dict(lr=1e-4),
    tasks_to_validate=[4])
SPLIT_MNIST["model_args"].update(
    kw_percent_on=0.05,
    weight_sparsity=0.8,
    hidden_sizes=[8192, 8192],
    output_size=NUM_CLASSES_PER_TASK,  # Single output head shared by all tasks
    num_segments=NUM_TASKS)
# ok yes they had a 10 hardcoded F.M.L.
# (dendrite_cl_experiment L84)
SPLIT_MNIST['train_model_args'] = dict(
    share_labels=True,
    num_labels=NUM_CLASSES_PER_TASK)

NUM_TASKS = 5
NUM_CLASSES_PER_TASK = 2
BATCH_SIZE = 512

SPLIT_CIFAR10 = deepcopy(BASE)
SPLIT_CIFAR10.update(
    experiment_class=SplitDataExperiment,
    dataset_class=b_datasets.SplitCIFAR10,
    batch_size=BATCH_SIZE,
    val_batch_size=BATCH_SIZE,
    num_tasks=NUM_TASKS,
    num_classes=NUM_CLASSES_PER_TASK * NUM_TASKS,
    epochs=5,
    optimizer_args=dict(lr=1e-4),
    tasks_to_validate=[4])
SPLIT_CIFAR10["model_args"].update(
    input_size=32 * 32 * 3,
    dim_context=32 * 32 * 3,
    output_size=NUM_CLASSES_PER_TASK,  # Single output head shared by all tasks
    kw_percent_on=0.05,
    weight_sparsity=0.8,
    hidden_sizes=[8192, 8192],
    num_segments=NUM_TASKS)
# ok yes they had a 10 hardcoded F.M.L.
# (dendrite_cl_experiment L84)
SPLIT_CIFAR10['train_model_args'] = dict(
    share_labels=True,
    num_labels=NUM_CLASSES_PER_TASK)
SPLIT_CIFAR10["dataset_args"].update(
    # TODO: figure out how to provide different transforms for train/test
    transform=transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]),
)

BATCH_SIZE = 512

SPLIT_CIFAR100_BASE = deepcopy(BASE)
SPLIT_CIFAR100_BASE.update(
    experiment_class=SplitDataExperiment,
    dataset_class=b_datasets.SplitCIFAR100,
    batch_size=BATCH_SIZE,
    val_batch_size=BATCH_SIZE,
    epochs=5,
    optimizer_args=dict(lr=1e-4),
    tasks_to_validate=[1, 4, 9, 24, 49, 99])
SPLIT_CIFAR100_BASE["model_args"].update(
    input_size=32 * 32 * 3,
    dim_context=32 * 32 * 3,
    kw_percent_on=0.05,
    weight_sparsity=0.8,
    hidden_sizes=[8192, 8192],
)
SPLIT_CIFAR100_BASE['train_model_args'] = dict(
    share_labels=True,
)
SPLIT_CIFAR100_BASE["dataset_args"].update(
    # TODO: figure out how to provide different transforms for train/test
    transform=transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]),
)


def make_cifar_config(num_tasks: int):
    num_classes_per_task = 100 // num_tasks

    config = deepcopy(SPLIT_CIFAR100_BASE)
    config.update(
        num_tasks=num_tasks,
        num_classes=num_classes_per_task * num_tasks,
    )
    config["model_args"].update(
        output_size=num_classes_per_task,  # Single output head shared by all tasks
        num_segments=num_tasks,
        kw_percent_on=0.05,
        weight_sparsity=0.8,
        hidden_sizes=[8192, 8192],
    )
    config['train_model_args'].update(
        num_labels=num_classes_per_task,
    )
    config["dataset_args"].update(
        num_tasks=num_tasks,
    )
    return config


SPLIT_CIFAR100_2 = make_cifar_config(num_tasks=2)
SPLIT_CIFAR100_5 = make_cifar_config(num_tasks=5)
SPLIT_CIFAR100_10 = make_cifar_config(num_tasks=10)

# Export configurations in this file
CONFIGS = dict(
    split_mnist=SPLIT_MNIST,
    split_cifar10=SPLIT_CIFAR10,
    split_cifar100_2=SPLIT_CIFAR100_2,
    split_cifar100_5=SPLIT_CIFAR100_5,
    split_cifar100_10=SPLIT_CIFAR100_10,
)
