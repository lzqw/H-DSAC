import copy
import datetime
import json
import os
import torch
import warnings

from common_utils import change_type, seed_everything

def init_args(**args):
    #args["algorithm"] = args['algorithm']
    # set torch parallel threads nums
    torch.set_num_threads(4)
    print("limit torch intra-op parallel threads num to {num} for saving computing resource.".format(num=4))
    # cuda
    if args["enable_cuda"]:
        if torch.cuda.is_available():
            args["use_gpu"] = True
        else:
            warning_msg = "cuda is not available, use CPU instead"
            warnings.warn(warning_msg)
            args["use_gpu"] = False
    else:
        args["use_gpu"] = False

    args["batch_size_per_sampler"] = args["sample_batch_size"]

