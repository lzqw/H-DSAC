import copy
import datetime
import json
import os
from initialization import create_alg,create_sampler
from common_utils import change_type
import torch
import warnings
import numpy as np
import yaml

OBS_DIM=250

class DPVPPolicy():
    def __init__(self):
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
            self.config = config
        self.config['obsv_dim']=OBS_DIM
        self.config['act_dim']=2
        self.config["action_high_limit"]=np.array([1.0, 1.0])
        self.config["action_low_limit"]=np.array([-1.0, -1.0])
        self.config["action_type"]="continu"
        if self.config["save_folder"] is None:
            dir_path = os.path.dirname(__file__)
            dir_path = os.path.dirname(dir_path)
            self.config["save_folder"] = os.path.join(
                dir_path + "/results/",
                self.config['algorithm'] +str("_")+ self.config['env_id'],
                datetime.datetime.now().strftime("%y%m%d-%H%M%S"),
            )
        os.makedirs(self.config["save_folder"], exist_ok=True)
        os.makedirs(self.config["save_folder"] + "/apprfunc", exist_ok=True)
        with open(self.config["save_folder"] + "/config.json", "w", encoding="utf-8") as f:
            json.dump(change_type(copy.deepcopy(self.config)), f, ensure_ascii=False, indent=4)

        self.dpvp = create_alg(**self.config)
        print("DPVP Policy is created.")
        
    