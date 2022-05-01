import os
from collections import Counter
from torchvision import datasets


def build_task_and_index_mappings(num_tasks, num_classes_per_task, targets):
    # task_to_targets[i] == list of all targets belonging to task `i`.
    task_to_targets = [
        tuple([num_classes_per_task * i + j for j in range(num_classes_per_task)])
        for i in range(num_tasks)]

    # index_to_task_id[i] == task ID assigned to the `i` example in the dataset.
    index_to_task_id = [
        target.item() // num_classes_per_task for target in targets]

    return task_to_targets, index_to_task_id


def check_dataset(ds):
    task_id_to_num_samples = Counter()
    for i in range(len(ds)):
        task_id = ds.get_task_id(i)
        task_id_to_num_samples[task_id] += 1

    for task_id in range(ds.num_tasks):
        assert task_id_to_num_samples[task_id] >= 1000, \
            f'[{task_id}] [{len(ds.data)}] {task_id_to_num_samples[task_id]}'


class GetTaskIDMixin:

    def get_task_id(self, index):
        assert 0 <= index < len(self.index_to_task_id), index
        res = self.index_to_task_id[index]
        assert isinstance(res, int), type(res)
        assert 0 <= res < self.num_tasks, res
        return res


class SplitMNIST(datasets.MNIST, GetTaskIDMixin):

    def __init__(self,
                 train,
                 num_tasks: int = 5,
                 root=".",
                 transform=None,
                 target_transform=None,
                 download=False):
        super().__init__(root=root, train=train, transform=transform,
                         target_transform=target_transform, download=download)

        self.num_tasks = num_tasks
        self.num_classes_per_task = len(self.classes) // self.num_tasks
        self.task_to_digits, self.index_to_task_id = build_task_and_index_mappings(
            self.num_tasks, self.num_classes_per_task, self.targets)
        check_dataset(self)

    @property
    def processed_folder(self):
        return os.path.join(self.root, "SplitMNIST", "processed")


class SplitCIFAR10(datasets.CIFAR10, GetTaskIDMixin):

    def __init__(self,
                 train,
                 num_tasks: int = 5,
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
        self.task_to_class_idx, self.index_to_task_id = build_task_and_index_mappings(
            self.num_tasks, self.num_classes_per_task, self.targets)
        check_dataset(self)

    @property
    def processed_folder(self):
        return os.path.join(self.root, "SplitCIFAR10", "processed")


class SplitCIFAR100(datasets.CIFAR100, GetTaskIDMixin):

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
        self.task_to_class_idx, self.index_to_task_id = build_task_and_index_mappings(
            self.num_tasks, self.num_classes_per_task, self.targets)
        check_dataset(self)

    @property
    def processed_folder(self):
        return os.path.join(self.root, "SplitCIFAR100", "processed")
