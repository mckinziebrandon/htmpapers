#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see htt"://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

"""
Experiment file that runs Active Dendrites Networks which 1) construct a prototype
context vector during training, and 2) try to infer the correct prototype for each task
during inference.
"""

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


class SplitMNIST(datasets.MNIST):

    def __init__(self, train, root=".", target_transform=None,
                 download=False, normalize=True):

        t = [transforms.ToTensor()]
        if normalize:
            t.append(transforms.Normalize((0.13062755,), (0.30810780,)))
        data_transform = transforms.Compose(t)
        super().__init__(root=root, train=train, transform=data_transform,
                         target_transform=target_transform, download=download)

        # (0, 1), (2, 3), (4, 5), (6, 7), (8, 9)
        self.num_tasks = 5
        self.task_to_digits = [
            (2 * i, 2 * i + 1) for i in range(self.num_tasks)]
        self.index_to_task_id = [
            self[i][1] // 2
            for i in range(len(self.data))
        ]

        from collections import Counter
        task_id_to_num_samples = Counter()
        for i in range(len(self)):
            task_id = self.get_task_id(i)
            task_id_to_num_samples[task_id] += 1

        for task_id in range(5):
            assert task_id_to_num_samples[task_id] > 1000, \
                f'[{task_id}] [{len(self.data)}] {task_id_to_num_samples[task_id]}'
            # assert task_id_to_num_samples[task_id] == len(self.data) // 5, \
            #     f'[{task_id}] [{len(self.data)}] {task_id_to_num_samples[task_id]}'

        # self.task_to_data = [[] for _ in range(self.num_tasks)]
        # for i in range(len(self.data)):
        #     img, target = super().__getitem__(i % len(self.data))
        #     task_idx = target // 2
        #     self.task_to_data[task_idx].append((img, target))

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, "SplitMNIST", "processed")

    def get_task_id(self, index):
        # return index % 2
        # return index % self.num_tasks
        assert 0 <= index < len(self.index_to_task_id), index
        res = self.index_to_task_id[index]
        assert isinstance(res, int), type(res)
        assert 0 <= res < 5, res
        return res

class SplitMNISTExperiment(vernon_mixins.RezeroWeights,
                          dendrites_mixins.PrototypeContext,
                          DendriteContinualLearningExperiment):
    pass


class PrototypeExperiment(vernon_mixins.RezeroWeights,
                          dendrites_mixins.PrototypeContext,
                          cl_mixins.PermutedMNISTTaskIndices,
                          DendriteContinualLearningExperiment):
    pass


class PrototypeFigure1BExperiment(dendrites_mixins.PrototypeFigure1B,
                                  dendrites_mixins.PlotHiddenActivations,
                                  PrototypeExperiment):
    pass


def seed_fn(spec):
    return np.random.randint(2, 10000)

PROTOTYPE_BASE = dict(
    experiment_class=PrototypeExperiment,
    num_samples=1,

    # Results path
    local_dir=os.path.expanduser("~/nta/results/experiments/dendrites"),

    dataset_class=PermutedMNIST,
    dataset_args=dict(
        root=os.path.expanduser("~/nta/results/data/"),
        download=True,  # Change to True if running for the first time
        seed=42,
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
SPLIT_MNIST = deepcopy(PROTOTYPE_BASE)
SPLIT_MNIST["dataset_class"] = SplitMNIST
SPLIT_MNIST["dataset_args"] = dict(
    root=os.path.expanduser("~/nta/results/data/"),
    download=True,  # Change to True if running for the first time
)
SPLIT_MNIST["model_args"].update(
    kw_percent_on=0.05,
    weight_sparsity=0.8,
    hidden_sizes=[8192, 8192],
    output_size=2,  # Single output head shared by all tasks
    num_segments=5)
SPLIT_MNIST.update(
    experiment_class=SplitMNISTExperiment,
    batch_size=512,
    val_batch_size=512,
    num_samples=3,
    num_tasks=5,
    num_classes=2 * 5,
    epochs=5,
    optimizer_args=dict(lr=1e-4),
    tasks_to_validate=[4],
)
# TODO: why the fuck is the required for me but not permutedMNIST???
# ok yes they had a 10 hardcoded F.M.L.
# (dendrite_cl_experiment L84)
SPLIT_MNIST['train_model_args'] = dict(
    share_labels=True,
    num_labels=2
)

PROTOTYPE_2 = deepcopy(PROTOTYPE_BASE)
PROTOTYPE_2["dataset_args"].update(num_tasks=2)
PROTOTYPE_2["model_args"].update(num_segments=2)
PROTOTYPE_2.update(
    num_tasks=2,
    num_classes=10 * 2,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=1,
    optimizer_args=dict(lr=5e-4),
)


PROTOTYPE_5 = deepcopy(PROTOTYPE_BASE)
PROTOTYPE_5["dataset_args"].update(num_tasks=5)
PROTOTYPE_5["model_args"].update(num_segments=5)
PROTOTYPE_5.update(
    num_tasks=5,
    num_classes=10 * 5,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=1,
    optimizer_args=dict(lr=5e-4),
)


PROTOTYPE_10 = deepcopy(PROTOTYPE_BASE)
PROTOTYPE_10["dataset_args"].update(num_tasks=10)
PROTOTYPE_10["model_args"].update(num_segments=10)
PROTOTYPE_10.update(
    num_tasks=10,
    num_classes=10 * 10,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=3,
    optimizer_args=dict(lr=5e-4),
)


# This experiment configuration is for visualizing the hidden activations in an Active
# Dendrites Network on a per-task basis; it produces `.pt` files which can then be used
# by the hidden activations script to generate visualizations
HIDDEN_ACTIVATIONS_PER_TASK = deepcopy(PROTOTYPE_10)
HIDDEN_ACTIVATIONS_PER_TASK.update(
    experiment_class=PrototypeFigure1BExperiment,

    plot_hidden_activations_args=dict(
        include_modules=[KWinners],
        plot_freq=1,
        max_samples_to_plot=5000
    ),
)


PROTOTYPE_25 = deepcopy(PROTOTYPE_BASE)
PROTOTYPE_25["dataset_args"].update(num_tasks=25)
PROTOTYPE_25["model_args"].update(num_segments=25)
PROTOTYPE_25.update(
    num_tasks=25,
    num_classes=10 * 25,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=5,
    optimizer_args=dict(lr=3e-4),
)


PROTOTYPE_50 = deepcopy(PROTOTYPE_BASE)
PROTOTYPE_50["dataset_args"].update(num_tasks=50)
PROTOTYPE_50["model_args"].update(num_segments=50)
PROTOTYPE_50.update(
    num_tasks=50,
    num_classes=10 * 50,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=3,
    optimizer_args=dict(lr=3e-4),
)


PROTOTYPE_100 = deepcopy(PROTOTYPE_BASE)
PROTOTYPE_100["dataset_args"].update(num_tasks=100)
PROTOTYPE_100["model_args"].update(num_segments=100)
PROTOTYPE_100.update(
    num_tasks=100,
    num_classes=10 * 100,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=3,
    optimizer_args=dict(lr=1e-4),
)


# Export configurations in this file
CONFIGS = dict(
    prototype_2=PROTOTYPE_2,
    prototype_5=PROTOTYPE_5,
    prototype_10=PROTOTYPE_10,
    hidden_activations_per_task=HIDDEN_ACTIVATIONS_PER_TASK,
    prototype_25=PROTOTYPE_25,
    prototype_50=PROTOTYPE_50,
    prototype_100=PROTOTYPE_100,
    split_mnist=SPLIT_MNIST,
)
