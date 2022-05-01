# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""
This module trains & evaluates a dendritic network in a continual learning setting on
permutedMNIST for a specified number of tasks/permutations. A context vector is
provided to the dendritic network, so task information need not be inferred.

This setup is very similar to that of context-dependent gating model from the paper
'Alleviating catastrophic forgetting using contextdependent gating and synaptic
stabilization' (Masse et al., 2018).
"""

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

import argparse
import copy

from experiments import CONFIGS
from nupic.research.frameworks.vernon.parser_utils import (
    get_default_parsers,
    process_args,
)
from nupic.research.frameworks.vernon.run import run as run
# from nupic.research.frameworks.ray.run_with_raytune import run as run_with_ray_tune
from nupic.research.frameworks.ray.run_with_raytune import (
    get_tune_kwargs,
    run_single_instance
)

def _run(config):
    if config.get("single_instance", False):
        assert False
        return run_single_instance(config)

    # Connect to ray
    local_mode = config.get("local_mode", False)
    if local_mode:
        assert False
        address = None
    else:
        address = os.environ.get("REDIS_ADDRESS", config.get("redis_address"))
    ray.init(address=address, local_mode=local_mode)

    # Register serializer and deserializer - needed when logging arrays and tensors.
    if not config.get("local_mode", False):
        register_torch_serializers()

    # Get ray.tune kwargs for the given config.
    kwargs = get_tune_kwargs(config)

    # Queue trials until the cluster scales up
    kwargs.update(queue_trials=False)

    pprint(kwargs)
    result = tune.run(**kwargs)
    ray.shutdown()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        parents=get_default_parsers(),
    )

    parser.add_argument("-e", "--experiment", dest="name", nargs="+",
                        default="default_base", help="Experiment to run",
                        choices=list(CONFIGS.keys()))
    parser.add_argument("--run_without_ray_tune", dest="run_without_ray_tune",
                        action='store_true',
                        help="run by calling ray.run_with_ray_tune or vernon.run")

    args = parser.parse_args()
    if args.name is None:
        parser.print_help()
        exit(1)

    for experiment in args.name:

        # Get configuration values
        config = copy.deepcopy(CONFIGS[experiment])

        # Merge configuration with command line arguments
        for k, v in vars(args).items():
            if k not in config:
                print(f'WARNING: Setting default CLI arg: config[{k}] = {v}')
                config[k] = v
            else:
                print(f'WARNING: Skipping CLI arg={k} with value={v} because it '
                      f'already exists in user config with value={config[k]}')
        config.update(name=experiment)

        config = process_args(args, config)

        if config is None:
            continue
        if args.run_without_ray_tune:
            run(config)
        else:
            _run(config)
            # run_with_ray_tune(config)
