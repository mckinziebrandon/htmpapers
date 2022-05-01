import abc

import numpy as np
import torch
from scipy.stats import f

from nupic.research.frameworks.dendrites import evaluate_dendrite_model
import torch
from nupic.research.frameworks.continual_learning.experiments import (
    ContinualLearningExperiment,
)
from nupic.research.frameworks.dendrites import (
    evaluate_dendrite_model,
    train_dendrite_model,
)

from nupic.research.frameworks.dendrites.dendrite_cl_experiment import (
    DendriteContinualLearningExperiment,
)
from nupic.research.frameworks.continual_learning import mixins as cl_mixins
from nupic.research.frameworks.dendrites import DendriticMLP
from nupic.research.frameworks.dendrites import mixins as dendrites_mixins
from nupic.research.frameworks.dendrites.dendrite_cl_experiment import (
    DendriteContinualLearningExperiment,
)
from torch.utils.data import DataLoader
from nupic.research.frameworks.pytorch.datasets import PermutedMNIST
from nupic.research.frameworks.vernon import mixins as vernon_mixins
from nupic.torch.modules import KWinners


# __all__ = [
#     "BrandonDendriteContinualLearningExperiment",
# ]


class BrandonDendriteContinualLearningExperiment(DendriteContinualLearningExperiment):

    def run_task(self):
        """
        Run the current task.
        """
        # configure the sampler to load only samples from current task
        self.logger.info("Training task %d...", self.current_task)
        self.train_loader.sampler.set_active_tasks(self.current_task)

        # Run epochs, inner loop
        # TODO: return the results from run_epoch
        self.current_epoch = 0
        for e in range(self.epochs):
            self.logger.info("Training task %d, epoch %d...", self.current_task, e)
            self.run_epoch()

        ret = {}
        total_correct = torch.tensor(0, device=self.device)
        total_tested = 0
        total_loss = torch.tensor(0., device=self.device)
        if self.current_task in self.tasks_to_validate:
            #  Compute metrics separately on each task.
            for task_idx in range(self.current_task + 1):
                self.val_loader.sampler.set_active_tasks(task_idx)
                task_metrics = self.validate()
                total_correct += task_metrics['total_correct']
                total_tested += task_metrics['total_tested']
                total_loss += task_metrics['mean_loss'] * task_metrics['total_tested']
                ret.update({
                    f'task_{task_idx}_{k}': v for k, v in task_metrics.items()
                })

            ret.update(dict(
                total_correct=total_correct.item(),
                total_tested=total_tested,
                mean_loss=total_loss.item() / total_tested if total_tested > 0 else 0,
                mean_accuracy=torch.true_divide(total_correct, total_tested).item() if total_tested > 0 else 0,
            ))

            self.val_loader.sampler.set_active_tasks(self.current_task)

        ret.update(
            learning_rate=self.get_lr()[0],
        )
        print(f"[task={self.current_task}] run_task ret: ", ret)

        self.current_task += 1

        if self.reset_optimizer_after_task:
            self.optimizer = self.recreate_optimizer(self.model)

        return ret

    def train_epoch(self):
        # (brandon): replaced hardcoded 10 with self.num_classes_per_task
        train_dendrite_model(
            model=self.model,
            loader=self.train_loader,
            optimizer=self.optimizer,
            device=self.device,
            criterion=self.error_loss,
            share_labels=True,
            num_labels=self.num_classes_per_task,
            context_vector=self.context_vector,
            train_context_fn=self.train_context_fn,
            post_batch_callback=self.post_batch_wrapper,
            batches_in_epoch=self.batches_in_epoch,
        )

    def validate(self, loader=None):
        """
        Run validation on the currently active tasks.

        Invoked within self.run_task(...) every `self.task_to_validate`.
        """
        if loader is None:
            loader = self.val_loader

        # (brandon): replaced hardcoded 10 with self.num_classes_per_task
        return evaluate_dendrite_model(
            model=self.model,
            loader=loader,
            device=self.device,
            criterion=self.error_loss,
            infer_context_fn=self.infer_context_fn,
            share_labels=True,
            num_labels=self.num_classes_per_task)

    @classmethod
    def load_dataset(cls, config, train=True):
        dataset_class = config.get("dataset_class", None)
        if dataset_class is None:
            raise ValueError("Must specify 'dataset_class' in config.")

        split = 'train' if train else 'valid'
        if f"{split}_dataset_args" in config:
            dataset_args = dict(config[f"{split}_dataset_args"])
        else:
            dataset_args = dict(config.get("dataset_args", {}))
        dataset_args.update(train=train)
        return dataset_class(**dataset_args)
