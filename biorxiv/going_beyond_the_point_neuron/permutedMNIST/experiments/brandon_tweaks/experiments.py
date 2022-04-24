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


class BrandonPrototypeContext(metaclass=abc.ABCMeta):

    def setup_experiment(self, config):
        # Since the prototype vector is an element-wise mean of individual data samples,
        # it's necessarily the same dimension as the input
        model_args = config.get("model_args")
        dim_context = model_args.get("dim_context")
        input_size = model_args.get("input_size")

        super().setup_experiment(config)

        prototype_context_args = config.get("prototype_context_args", {})
        self.construct = prototype_context_args.get("construct", False)

        # Tensor for accumulating each task's prototype vector
        self.contexts = torch.zeros((0, self.model.dim_context))
        self.contexts = self.contexts.to(self.device)

        if self.construct:

            # Store "exemplars" for each context vector as a list of Torch Tensors;
            # these are used to perform statistical tests against a new batch of data to
            # determine if that new batch corresponds to the same task
            self.clusters = []

            # `contexts` needs to be a mutable data type in order to be modified by a
            # nested function (below), so it is simply wrapped in a 1-element list
            self.contexts = [self.contexts]

            # This list keeps track of how many exemplars have been used to compute each
            # context vector since 1) we compute a weighted average, and 2) most
            # exemplars are discarded for memory efficiency
            self.contexts_n = []

            # In order to perform statistical variable transformations (below), there
            # are restrictions on the dimensionality of the input, so subindices
            # randomly sample features and discard others
            self.subindices = np.random.choice(range(input_size), size=dim_context,
                                               replace=False)
            self.subindices.sort()

        else:

            # Since the prototype vector is an element-wise mean of individual data
            # samples it's necessarily the same dimension as the input
            assert dim_context == input_size, ("For prototype experiments `dim_context`"
                                               " must match `input_size`")

    def run_task(self):
        self.train_loader.sampler.set_active_tasks(self.current_task)

        if self.construct:

            self.train_context_fn = dendrites_mixins.prototype_context.construct_prototype(
                self.clusters, self.contexts, self.contexts_n, self.subindices)

        else:

            # Find a context vector by computing the prototype of all training examples
            self.context_vector = dendrites_mixins.prototype_context.compute_prototype(
                self.train_loader).to(self.device)
            self.contexts = torch.cat((
                self.contexts, self.context_vector.unsqueeze(0)))
            self.train_context_fn = dendrites_mixins.prototype_context.train_prototype(
                self.context_vector)

        return super().run_task()

    def validate(self, loader=None):
        if loader is None:
            loader = self.val_loader

        if self.construct:
            infer_context_fn = dendrites_mixins.prototype_context.infer_prototype(
                self.contexts[0], self.subindices)
        else:
            infer_context_fn = dendrites_mixins.prototype_context.infer_prototype(
                self.contexts)

        # (brandon): replaced hardcoded 10 with self.num_classes_per_task
        return evaluate_dendrite_model(model=self.model,
                                       loader=loader,
                                       device=self.device,
                                       criterion=self.error_loss,
                                       share_labels=True,
                                       num_labels=self.num_classes_per_task,
                                       infer_context_fn=infer_context_fn)
