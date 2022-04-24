
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

from brandon_tweaks import (
    experiments as brandon_experiments,
    datasets as brandon_datasets,
)


class SplitMNISTExperiment(vernon_mixins.RezeroWeights,
                           brandon_experiments.BrandonPrototypeContext,
                           brandon_experiments.BrandonDendriteContinualLearningExperiment):
    pass


def seed_fn(spec):
    return np.random.randint(2, 10000)

SPLIT_MNIST_BASE = dict(
    experiment_class=SplitMNISTExperiment,
    num_samples=1,

    # Results path
    local_dir=os.path.expanduser("~/nta/results/experiments/dendrites"),

    dataset_class=brandon_datasets.SplitMNIST,
    dataset_args=dict(
        root=os.path.expanduser("~/nta/results/data/"),
        download=True,  # Change to True if running for the first time
    ),

    model_class=DendriticMLP,
    model_args=dict(
        input_size=784,
        output_size=10,  # Single output head shared by all tasks
        hidden_sizes=[2048, 2048],
        dim_context=784,
        kw=True,
        kw_percent_on=0.05,
        dendrite_weight_sparsity=0.0,
        weight_sparsity=0.5,
        context_percent_on=0.1,
    ),

    # batch_size=2,
    # val_batch_size=2,
    # tasks_to_validate=[4],
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
SPLIT_MNIST = deepcopy(SPLIT_MNIST_BASE)

NUM_TASKS = 5
NUM_CLASSES_PER_TASK = 2
BATCH_SIZE = 512

SPLIT_MNIST["model_args"].update(
    kw_percent_on=0.05,
    weight_sparsity=0.8,
    hidden_sizes=[8192, 8192],
    output_size=NUM_CLASSES_PER_TASK,  # Single output head shared by all tasks
    num_segments=NUM_TASKS)
SPLIT_MNIST.update(
    experiment_class=SplitMNISTExperiment,
    batch_size=BATCH_SIZE,
    val_batch_size=BATCH_SIZE,
    num_samples=3,
    num_tasks=NUM_TASKS,
    num_classes=NUM_CLASSES_PER_TASK * NUM_TASKS,
    epochs=5,
    optimizer_args=dict(lr=1e-4),
    tasks_to_validate=[4],
)
# ok yes they had a 10 hardcoded F.M.L.
# (dendrite_cl_experiment L84)
SPLIT_MNIST['train_model_args'] = dict(
    share_labels=True,
    num_labels=NUM_CLASSES_PER_TASK
)

# Export configurations in this file
CONFIGS = dict(
    split_mnist=SPLIT_MNIST,
)
