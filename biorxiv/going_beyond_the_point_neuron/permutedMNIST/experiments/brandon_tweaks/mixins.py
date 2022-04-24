import math
from collections import defaultdict


class SplitDatasetTaskIndices:
    @classmethod
    def compute_task_indices(cls, config, dataset):
        # Defines how many classes should exist per task
        num_tasks = config["num_tasks"]
        num_classes = config.get("num_classes", None)
        assert num_classes is not None, "num_classes should be defined"
        num_classes_per_task = math.floor(num_classes / num_tasks)

        # Assume dataloaders are already created
        class_indices = defaultdict(list)

        # This for-loop is the only difference bw here and ContinualLearningExperiment.compute_task_indices.
        # Faster to loop over idx than the full (img, targets) tuples.
        for idx, target in enumerate(dataset.targets):
            class_indices[int(target)].append(idx)

        task_indices = defaultdict(list)
        for i in range(num_tasks):
            for j in range(num_classes_per_task):
                task_indices[i].extend(class_indices[j + (i * num_classes_per_task)])
        return task_indices
