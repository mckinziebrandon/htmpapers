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
from nupic.research.frameworks.pytorch.datasets import PermutedMNIST
from nupic.research.frameworks.vernon import mixins as vernon_mixins
from nupic.torch.modules import KWinners


__all__ = [
    "BrandonDendriteContinualLearningExperiment",
]


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


class BrandonPrototypeContext(dendrites_mixins.PrototypeContext):

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
