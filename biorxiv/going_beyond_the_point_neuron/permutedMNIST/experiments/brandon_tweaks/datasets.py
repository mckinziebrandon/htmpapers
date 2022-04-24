from collections import Counter
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

        task_id_to_num_samples = Counter()
        for i in range(len(self)):
            task_id = self.get_task_id(i)
            task_id_to_num_samples[task_id] += 1

        for task_id in range(5):
            assert task_id_to_num_samples[task_id] > 1000, \
                f'[{task_id}] [{len(self.data)}] {task_id_to_num_samples[task_id]}'

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


class SplitCIFAR10(datasets.CIFAR10):

    def __init__(self,
                 train,
                 root=".",
                 transform=None,
                 target_transform=None,
                 download=False):
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download)

        self.num_tasks = 5
        self.task_to_class_idx = [
            (2 * i, 2 * i + 1) for i in range(self.num_tasks)]
        self.index_to_task_id = [
            self[i][1] // 2 for i in range(len(self.data))]

        task_id_to_num_samples = Counter()
        for i in range(len(self)):
            task_id = self.get_task_id(i)
            task_id_to_num_samples[task_id] += 1

        for task_id in range(5):
            assert task_id_to_num_samples[task_id] > 1000, \
                f'[{task_id}] [{len(self.data)}] {task_id_to_num_samples[task_id]}'

    @property
    def processed_folder(self):
        return os.path.join(self.root, "SplitCIFAR10", "processed")

    def get_task_id(self, index):
        assert 0 <= index < len(self.index_to_task_id), index
        res = self.index_to_task_id[index]
        assert isinstance(res, int), type(res)
        assert 0 <= res < 5, res
        return res


class SplitCIFAR100(datasets.CIFAR100):

    def __init__(self,
                 train,
                 num_tasks: int,
                 root=".",
                 transform=None,
                 target_transform=None,
                 download=False):
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download)

        self.num_tasks = num_tasks
        self.num_classes_per_task = len(self.classes) // self.num_tasks
        self.task_to_class_idx = [
            tuple([2 * i + j for j in range(self.num_classes_per_task)])
            for i in range(self.num_tasks)]
        self.index_to_task_id = [
            self.targets[i] // self.num_classes_per_task for i in range(len(self.data))]

        task_id_to_num_samples = Counter()
        for i in range(len(self)):
            task_id = self.get_task_id(i)
            task_id_to_num_samples[task_id] += 1

        for task_id in range(self.num_tasks):
            assert task_id_to_num_samples[task_id] >= 1000, \
                f'[{task_id}] [{len(self.data)}] {task_id_to_num_samples[task_id]}'

    @property
    def processed_folder(self):
        return os.path.join(self.root, "SplitCIFAR100", "processed")

    def get_task_id(self, index):
        assert 0 <= index < len(self.index_to_task_id), index
        res = self.index_to_task_id[index]
        assert isinstance(res, int), type(res)
        return res
