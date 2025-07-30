import time

import numpy as np
import torch

from utils.initialization import create_env
from utils.common_utils import set_seed
from utils.explore_noise import GaussNoise, EpsilonGreedy
from utils.tensorboard_setup import tb_tags

__all__ = ["PvpSampler"]

class PvpSampler:
    def __init__(self, index=0, **kwargs):
        # initialize necessary hyperparameters
        self.env = create_env(**kwargs)
        _, self.env = set_seed(kwargs["trainer"], kwargs["seed"], index + 200, self.env)
        self.obs, self.info = self.env.reset()
        self.has_render = hasattr(self.env, "render")
        self.total_cost=0
        alg_name = kwargs["algorithm"]
        alg_file_name = alg_name.lower()
        file = __import__(alg_file_name)
        ApproxContainer = getattr(file, "ApproxContainer")
        self.networks = ApproxContainer(**kwargs)
        self.noise_params = kwargs["noise_params"]
        self.sample_batch_size = kwargs["batch_size_per_sampler"]
        self.policy_func_name = kwargs["policy_func_name"]
        self.action_type = kwargs["action_type"]
        self.obsv_dim = kwargs["obsv_dim"]
        self.act_dim = kwargs["action_dim"]
        self.discard_reward = kwargs["discard_reward"]
        self.takeover_stop_td=kwargs["takeover_stop_td"]
        self.total_sample_number = 0
        self.reward_scale = 1.0
        self.reward_list=[]
        self.ep_reward=0

        if self.noise_params is not None:
            if self.action_type == "continu":
                self.noise_processor = GaussNoise(**self.noise_params)
            elif self.action_type == "discret":
                self.noise_processor = EpsilonGreedy(**self.noise_params)

    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)

    def sample(self):
        self.total_sample_number += self.sample_batch_size
        tb_info = dict()
        start_time = time.perf_counter()
        batch_data = []
        human_batch_data = []

        self.episode = {}
        self.episode["velocity"] = []
        self.episode["steering"] = []
        self.episode["step_reward"] = []
        self.episode["acceleration"] = []
        self.episode["cost"] = []

        for _ in range(self.sample_batch_size):
            # take action using behavior policy
            batch_obs = torch.from_numpy(
                np.expand_dims(self.obs, axis=0).astype("float32")
            )

            if not self.networks.activate_rl:
                logits = self.networks.policy(batch_obs)

                action_distribution = self.networks.create_action_distributions(logits)
                action, logp = action_distribution.sample()
                action = action.detach()[0].numpy()
                logp = logp.detach()[0].numpy()

                use_rl=False
                activate_rl=False

            else:
                activate_rl=True
                logits = self.networks.policy(batch_obs)
                action_distribution = self.networks.create_action_distributions(logits)
                action, logp = action_distribution.sample()
                action = action_distribution.mode()
                StochaQ=self.networks.bq1(batch_obs,action)
                mean,std=StochaQ[...,0],StochaQ[...,-1]
                action = action.detach()[0].numpy()


                logits = self.networks.policy_rl(batch_obs)
                action_distribution = self.networks.create_action_distributions(logits, True)
                action_rl, logp_rl = action_distribution.sample()
                StochaQ_rl=self.networks.bq1(batch_obs,action_rl)
                mean_rl,std_rl=StochaQ_rl[...,0],StochaQ_rl[...,-1]
                action_rl = action_rl.detach()[0].numpy()
                logp_rl = logp_rl.detach()[0].numpy()

                out=(mean_rl-mean)/(((std_rl)**2+(std)**2)**0.5)
                if out>=0.8:
                    use_rl=True
                    action=action_rl
                    logp=logp_rl
                else:
                    use_rl=False
                    action=action
                    logp=logp

            action = np.array(action)
            if self.action_type == "continu":
                action_clip = action.clip(
                    self.env.action_space.low, self.env.action_space.high
                )
            else:
                action_clip = action
            # interact with environment
            self.env.env.env.env.env.use_rl=use_rl
            self.env.env.env.env.env.activate_rl=activate_rl
            next_obs, reward, self.done, next_info = self.env.step(action_clip)
            if "TimeLimit.truncated" not in next_info.keys():
                next_info["TimeLimit.truncated"] = False
            if next_info["TimeLimit.truncated"]:
                self.done = False

            intervention=next_info['takeover']
            intervention_start=next_info['takeover_start']
            intervention_cost=next_info['takeover_cost']
            action_behavior=next_info['raw_action']


            action_novice=action

            if self.discard_reward:
                reward = 0.0
            else:
                reward = reward


            if self.takeover_stop_td:
                _stop_td = intervention
            else:
                _stop_td = intervention_start

            data = [
                self.obs.copy(),
                self.info,
                action_behavior,
                self.reward_scale * reward,
                next_obs.copy(),
                self.done,
                logp,
                next_info,

                action_novice,
                intervention,
                intervention_start,
                np.float32(intervention_cost),
                np.float32(1-_stop_td),

                np.float32(activate_rl),
                np.float32(use_rl),
            ]

            batch_data.append(tuple(data))
            self.reward_list.append(reward)
            if self.done:
                self.ep_reward=sum(self.reward_list)
                self.reward_list=[]
            self.episode["velocity"].append(next_info.get("velocity", 0))
            self.episode["steering"].append(next_info.get("steering", 0))
            self.episode["step_reward"].append(next_info.get("step_reward", 0))
            self.episode["acceleration"].append(next_info.get("acceleration", 0))
            self.episode["cost"].append(next_info.get("cost", 0))
            self.total_cost += next_info.get("cost", 0)

            self.obs = next_obs
            self.info = next_info
            if self.done or next_info["TimeLimit.truncated"]:
                self.obs, self.info = self.env.reset()

        end_time = time.perf_counter()
        take_over_rate=np.mean(np.array(self.env.env.env.env.env.takeover_recorder) * 100)
        tb_info["takeover_rate"]=take_over_rate
        tb_info[tb_tags["sampler_time"]] = (end_time - start_time) * 1000

        tb_info["velocity_max"] = float(np.max(self.episode["velocity"]))
        tb_info["velocity_mean"] = float(np.mean(self.episode["velocity"]))
        tb_info["velocity_min"] = float(np.min(self.episode["velocity"]))
        tb_info["steering_max"] = float(np.max(self.episode["steering"]))
        tb_info["steering_mean"] = float(np.mean(self.episode["steering"]))
        tb_info["steering_min"] = float(np.min(self.episode["steering"]))
        tb_info["acceleration_min"] = float(np.min(self.episode["acceleration"]))
        tb_info["acceleration_mean"] = float(np.mean(self.episode["acceleration"]))
        tb_info["acceleration_max"] = float(np.max(self.episode["acceleration"]))
        tb_info["step_reward_max"] = float(np.max(self.episode["step_reward"]))
        tb_info["step_reward_mean"] = float(np.mean(self.episode["step_reward"]))
        tb_info["step_reward_min"] = float(np.min(self.episode["step_reward"]))
        tb_info["ep_reward"]=float(self.ep_reward)
        tb_info["ep_reward"]=float(self.ep_reward)

        tb_info["cost"] = float(sum(self.episode["cost"]))
        tb_info["total_cost"] = float(self.total_cost)


        return batch_data, tb_info

    def get_total_sample_number(self):
        return self.total_sample_number

