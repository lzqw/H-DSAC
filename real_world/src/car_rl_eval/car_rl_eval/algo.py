import time
from copy import deepcopy
from typing import Tuple

import numpy as np

import torch
import torch.nn as nn
from gym.envs.classic_control.acrobot import bound
from torch.distributions import Normal
from torch.optim import Adam
from common_utils import get_apprfunc_dict

from typing import Dict


class ApproxContainer(torch.nn.Module):
    """Approximate function container for DSAC_V2.

    Contains one policy and one action value.
    """

    def __init__(self, **kwargs):
        super().__init__()

        q_args = get_apprfunc_dict("value", kwargs["value_func_type"], **kwargs)
        self.q1: nn.Module = create_apprfunc(**q_args)
        self.q2: nn.Module = create_apprfunc(**q_args)
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)

        