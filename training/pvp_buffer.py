import numpy as np
import sys
import torch
from utils.common_utils import set_seed
import pickle
__all__ = ["PvpBuffer"]


def combined_shape(length: int, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class PvpBuffer:
    """
    Implementation of replay buffer with uniform sampling probability.
    """

    def __init__(self, index=0, **kwargs):
        set_seed(kwargs["trainer"], kwargs["seed"], index + 100)
        self.obsv_dim = kwargs["obsv_dim"]
        self.act_dim = kwargs["action_dim"]
        self.max_size = kwargs["buffer_max_size"]
        self.save_buffer=kwargs['save_buffer']
        self.save_budffer_fre=kwargs['save_buffer_fre']
        self.save_folder = kwargs["save_folder"]

        if kwargs['load_buffer']:
            self.load(path=kwargs['buffer_path'])
        else:
            self.buf = {
                "obs": np.zeros(
                    combined_shape(self.max_size, self.obsv_dim), dtype=np.float32
                ),
                "obs2": np.zeros(
                    combined_shape(self.max_size, self.obsv_dim), dtype=np.float32
                ),
                "action_behavior": np.zeros(
                    combined_shape(self.max_size, self.act_dim), dtype=np.float32
                ),
                "rew": np.zeros(self.max_size, dtype=np.float32),
                "done": np.zeros(self.max_size, dtype=np.float32),
                "logp": np.zeros(self.max_size, dtype=np.float32),

                "intervention":np.zeros(self.max_size, dtype=np.float32),
                "intervention_start": np.zeros(self.max_size, dtype=np.float32),
                "intervention_cost": np.zeros(self.max_size, dtype=np.float32),
                "action_novice": np.zeros(
                    combined_shape(self.max_size, self.act_dim), dtype=np.float32
                ),
                "stop_td": np.zeros(self.max_size, dtype=np.float32),
                "activate_rl": np.zeros(self.max_size, dtype=np.float32),
                "use_rl": np.zeros(self.max_size, dtype=np.float32),
            }
            self.ptr, self.size, = (
                0,
                0,
            )

        if kwargs['load_human_buffer']:
            self.load_human(path=kwargs['human_buffer_path'])
        else:
            self.human_buf = {
                "obs": np.zeros(
                    combined_shape(self.max_size, self.obsv_dim), dtype=np.float32
                ),
                "obs2": np.zeros(
                    combined_shape(self.max_size, self.obsv_dim), dtype=np.float32
                ),
                "action_behavior": np.zeros(
                    combined_shape(self.max_size, self.act_dim), dtype=np.float32
                ),
                "rew": np.zeros(self.max_size, dtype=np.float32),
                "done": np.zeros(self.max_size, dtype=np.float32),
                "logp": np.zeros(self.max_size, dtype=np.float32),

                "intervention":np.zeros(self.max_size, dtype=np.float32),
                "intervention_start": np.zeros(self.max_size, dtype=np.float32),
                "intervention_cost": np.zeros(self.max_size, dtype=np.float32),
                "action_novice": np.zeros(
                    combined_shape(self.max_size, self.act_dim), dtype=np.float32
                ),
                "stop_td": np.zeros(self.max_size, dtype=np.float32),
                "activate_rl": np.zeros(self.max_size, dtype=np.float32),
                "use_rl": np.zeros(self.max_size, dtype=np.float32),
            }
            self.human_ptr, self.human_size, = (
                0,
                0,
            )
        self.discard_reward = kwargs["discard_reward"]
        if not self.discard_reward:
            print("You are not discarding reward from the environment! This should be True when training HACO!")



    def __len__(self):
        return self.size

    def save(self, filepath):
        buf = {
            "buf": self.buf,
            "ptr": self.ptr,
            "size": self.size,
        }

        with open(filepath+'/buf_'+str(self.size)+'.pkl', 'wb') as f:
            pickle.dump(buf, f)

    def save_human(self, filepath):
        human_buf = {
            "buf": self.human_buf,
            "ptr": self.human_ptr,
            "size": self.human_size,
        }
        with open(filepath+'/human_buf_'+str(self.human_size)+'.pkl', 'wb') as f:
            pickle.dump(human_buf, f)

    def load_human(self, path):
        with open(path, 'rb') as f:
            human_data = pickle.load(f)
            self.human_buf = human_data["buf"]
            self.human_ptr = human_data["ptr"]
            self.human_size = human_data["size"]

            # 扩充到最大长度 max_size
            for key in self.human_buf:
                current_shape = self.human_buf[key].shape
                if current_shape[0] < self.max_size:
                    # 创建扩展部分并填充
                    padding_shape = (self.max_size - current_shape[0], *current_shape[1:])
                    padding = np.zeros(padding_shape, dtype=self.human_buf[key].dtype)
                    self.human_buf[key] = np.concatenate([self.human_buf[key], padding], axis=0)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.buf = data["buf"]
            self.ptr = data["ptr"]
            self.size = data["size"]

            # 扩充到最大长度 max_size
            for key in self.buf:
                current_shape = self.buf[key].shape
                if current_shape[0] < self.max_size:
                    # 创建扩展部分并填充
                    padding_shape = (self.max_size - current_shape[0], *current_shape[1:])
                    padding = np.zeros(padding_shape, dtype=self.buf[key].dtype)
                    self.buf[key] = np.concatenate([self.buf[key], padding], axis=0)


    def __get_RAM__(self):
        return int(sys.getsizeof(self.buf)) * self.size / (self.max_size * 1000000)+int(sys.getsizeof(self.human_buf)) * self.human_size / (self.max_size * 1000000)

    def store(
        self,
        obs: np.ndarray,
        info: dict,
        action_behavior: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
        logp: np.ndarray,
        next_info: dict,


        action_novice: np.ndarray,
        intervention: np.ndarray,
        intervention_start: np.ndarray,
        intervention_cost: np.ndarray,
        stop_ed: np.ndarray,
        activate_rl: np.ndarray,
        use_rl: np.ndarray,
    ):
        if next_info['takeover'] or next_info['takeover_start']:
            self.human_buf["obs"][self.human_ptr] = obs
            self.human_buf["obs2"][self.human_ptr] = next_obs
            self.human_buf["action_behavior"][self.human_ptr] = action_behavior
            self.human_buf["rew"][self.human_ptr] = rew
            self.human_buf["done"][self.human_ptr] = done
            self.human_buf["logp"][self.human_ptr] = logp

            self.human_buf["action_novice"][self.human_ptr] = action_novice
            self.human_buf["intervention"][self.human_ptr] = intervention
            self.human_buf["intervention_start"][self.human_ptr] = intervention_start
            self.human_buf["intervention_cost"][self.human_ptr] = intervention_cost
            self.human_buf["stop_td"][self.human_ptr] = stop_ed

            self.human_buf["activate_rl"][self.human_ptr] = activate_rl
            self.human_buf["use_rl"][self.human_ptr] = use_rl

            # for k in self.additional_info.keys():
            #     self.human_buf[k][self.human_ptr] = info[k]
            #     self.human_buf["next_" + k][self.human_ptr] = next_info[k]
            self.human_ptr = (self.human_ptr + 1) % self.max_size
            self.human_size = min(self.human_size + 1, self.max_size)
            if self.save_buffer and self.human_size % self.save_budffer_fre == 0:
                print("save human buffer:",self.human_size)
                self.save_human(self.save_folder)

        else:
            self.buf["obs"][self.ptr] = obs
            self.buf["obs2"][self.ptr] = next_obs
            self.buf["action_behavior"][self.ptr] = action_behavior
            self.buf["rew"][self.ptr] = rew
            self.buf["done"][self.ptr] = done
            self.buf["logp"][self.ptr] = logp

            self.buf["action_novice"][self.ptr] = action_novice
            self.buf["intervention"][self.ptr] = intervention
            self.buf["intervention_start"][self.ptr] = intervention_start
            self.buf["intervention_cost"][self.ptr] = intervention_cost
            self.buf["stop_td"][self.ptr] = stop_ed

            self.buf["activate_rl"][self.ptr] = activate_rl
            self.buf["use_rl"][self.ptr] = use_rl

            # for k in self.additional_info.keys():
            #     self.buf[k][self.ptr] = info[k]
            #     self.buf["next_" + k][self.ptr] = next_info[k]
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

            if self.save_buffer and self.size % self.save_budffer_fre == 0 and self.size!=self.max_size:
                print("save buffer:", self.size)
                self.save(self.save_folder)

    def add_batch(self, samples: list):
        for sample in samples:
            self.store(*sample)

    def sample_batch(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = {}
        for k, v in self.buf.items():
            batch[k] = v[idxs]
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def sample_human_batch(self, batch_size: int):
        idxs = np.random.randint(0, self.human_size, size=batch_size)
        batch = {}
        for k, v in self.human_buf.items():
            batch[k] = v[idxs]
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

