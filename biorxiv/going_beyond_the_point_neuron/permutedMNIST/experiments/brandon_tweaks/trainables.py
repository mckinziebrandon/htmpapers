
import ray
import ray.resource_spec
import ray.util.sgd.utils as ray_utils
from ray.exceptions import RayActorError
from ray.tune import Trainable
from ray.tune.resources import Resources
from ray.tune.result import DONE, RESULT_DUPLICATE
from ray.tune.utils import warn_if_slow

import os
import time
from pprint import pprint

import ray
import ray.resource_spec
import torch
from ray.tune import Trainable, tune

from nupic.research.frameworks.vernon import interfaces
from nupic.research.frameworks.vernon.experiment_utils import get_free_port
from nupic.research.frameworks.vernon.search import TrialsCollection

from nupic.research.frameworks.ray.ray_utils import get_last_checkpoint, register_torch_serializers
from nupic.research.frameworks.ray.trainables import DistributedTrainable, RemoteProcessTrainable
from ray.tune.resources import Resources


class BrandonRemoteProcessTrainable(RemoteProcessTrainable):

    @classmethod
    def default_resource_request(cls, config):
        """
        Configure the cluster resources used by this experiment
        """
        num_cpus = max(config.get("num_cpus", 1),
                       config.get("workers", 0))
        num_gpus = config.get('num_gpus', 0)
        # Reminder: the cpu/gpu below mean for the ray task itself,
        # but I guess the trainable (re: my code) is represented as the `extra_` stuff?
        # https://docs.ray.io/en/releases-0.8.7/tune/api_docs/trainable.html?highlight=extra_gpu#advanced-resource-allocation
        return Resources(
            cpu=1,
            gpu=0,
            extra_cpu=num_cpus,
            extra_gpu=num_gpus,
            extra_memory=config.get('memory', 0))
