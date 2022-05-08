from .configs import CONFIGS as BRANDON_CONFIGS
from .tweaks import CONFIGS as TWEAKS_CONFIGS

__all__ = ["CONFIGS"]

# Collect all configurations
CONFIGS = dict()
CONFIGS.update(BRANDON_CONFIGS)
CONFIGS.update(TWEAKS_CONFIGS)
