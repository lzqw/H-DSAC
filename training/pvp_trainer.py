__all__ = ["PvpTrainer"]

from cmath import inf
import os
import time

import numpy as np
import torch
from scipy.optimize import brenth
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.tensorboard_setup import add_scalars
from utils.tensorboard_setup import tb_tags
from utils.common_utils import ModuleOnDevice


class PvpTrainer:
    def __init__(self, alg, sampler, buffer, evaluator, **kwargs):
        self.alg = alg
        self.sampler = sampler
        self.buffer = buffer
        self.per_flag = kwargs["buffer_name"] == "prioritized_replay_buffer"
        self.evaluator = evaluator
        self.use_eval = kwargs["use_eval"]

        # create center network
        self.networks = self.alg.networks
        self.sampler.networks = self.networks
        self.evaluator.networks = self.networks

        # initialize center network
        if kwargs["ini_network_dir"] is not None:
            self.load_apprfunc(kwargs["ini_network_dir"])
            if "ini_network_iter" not in kwargs:
                raise KeyError("The key 'ini_network_iter' is missing from kwargs")
            self.iteration = kwargs["ini_network_iter"]
            # self.networks.log_alpha=nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

        else:
            self.iteration = 0

        self.replay_batch_size = kwargs["replay_batch_size"]
        self.max_iteration = kwargs["max_iteration"]
        self.sample_interval = kwargs.get("sample_interval", 1)
        self.log_save_interval = kwargs["log_save_interval"]
        self.apprfunc_save_interval = kwargs["apprfunc_save_interval"]
        self.eval_interval = kwargs["eval_interval"]
        self.best_tar = -inf
        self.save_folder = kwargs["save_folder"]


        self.writer = SummaryWriter(log_dir=self.save_folder, flush_secs=20)
        # flush tensorboard at the beginning
        add_scalars(
            {tb_tags["alg_time"]: 0, tb_tags["sampler_time"]: 0}, self.writer, 0
        )
        self.writer.flush()


        while self.buffer.size < kwargs["buffer_warm_size"] and self.buffer.human_size < kwargs["buffer_warm_size"]:
            samples, _ = self.sampler.sample()
            self.buffer.add_batch(samples)

        self.use_gpu = kwargs["use_gpu"]
        if self.use_gpu:
            self.networks.cuda()

        self.start_time = time.time()
        if kwargs['warm_up']:
            self.warm_up(kwargs["warm_up_step"])

    def warm_up(self,warm_up_steps):
        print("start warm up")
        pbar=tqdm(total=warm_up_steps)
        while self.iteration < warm_up_steps:
            if self.buffer.human_size > self.replay_batch_size and self.buffer.size > self.replay_batch_size:
                replay_samples_agent= self.buffer.sample_batch(int(self.replay_batch_size/2))
                replay_samples_human = self.buffer.sample_human_batch(int(self.replay_batch_size/2))
                replay_samples = {k: torch.cat((replay_samples_agent[k], replay_samples_human[k]), dim=0) for k in
                                  replay_samples_agent.keys()}
            elif self.buffer.human_size > self.replay_batch_size:
                replay_samples = self.buffer.sample_human_batch(self.replay_batch_size)
            elif self.buffer.size > self.replay_batch_size:
                replay_samples = self.buffer.sample_batch(self.replay_batch_size)
            else:
                raise ValueError("Buffer size is too small to warmup")

            # learning
            if self.use_gpu:
                for k, v in replay_samples.items():
                    replay_samples[k] = v.cuda()

            if self.per_flag:
                alg_tb_dict, idx, new_priority = self.alg.local_update(
                    replay_samples, self.iteration
                )
                self.buffer.update_batch(idx, new_priority)
            else:
                alg_tb_dict = self.alg.local_update(replay_samples, self.iteration)

            # log
            if self.iteration % self.log_save_interval == 0:
                print("Iter = ", self.iteration)
                add_scalars(alg_tb_dict, self.writer, step=self.iteration)

            # save
            if self.iteration % self.apprfunc_save_interval == 0:
                self.save_apprfunc()

            self.iteration += 1
            pbar.update(1)
        print("warm up finished")
        self.save_apprfunc()
        with ModuleOnDevice(self.networks, "cpu"):
            total_avg_return = self.evaluator.run_evaluation(self.iteration, self.sampler.env)

    def step(self):
        # sampling
        sampler_tb_dict = {}
        if self.iteration % self.sample_interval == 0:
            with ModuleOnDevice(self.networks, "cpu"):
                sampler_samples, sampler_tb_dict = self.sampler.sample()
            self.buffer.add_batch(sampler_samples)

        if self.networks.activate_rl:
            replay_samples = self.buffer.sample_batch(self.replay_batch_size)
        else:
            # replay
            if self.buffer.human_size > self.replay_batch_size and self.buffer.size > self.replay_batch_size:
                replay_samples_agent= self.buffer.sample_batch(int(self.replay_batch_size/2))
                replay_samples_human = self.buffer.sample_human_batch(int(self.replay_batch_size/2))
                replay_samples = {k: torch.cat((replay_samples_agent[k], replay_samples_human[k]), dim=0) for k in
                                  replay_samples_agent.keys()}
            elif self.buffer.human_size > self.replay_batch_size:
                replay_samples = self.buffer.sample_human_batch(self.replay_batch_size)
            elif self.buffer.size > self.replay_batch_size:
                replay_samples = self.buffer.sample_batch(self.replay_batch_size)
            else:
                return

        # learning
        if self.use_gpu:
            for k, v in replay_samples.items():
                replay_samples[k] = v.cuda()

        if self.per_flag:
            alg_tb_dict, idx, new_priority = self.alg.local_update(
                replay_samples, self.iteration
            )
            self.buffer.update_batch(idx, new_priority)
        else:
            alg_tb_dict = self.alg.local_update(replay_samples, self.iteration)
        # log
        if self.iteration % self.log_save_interval == 0:
            print("Iter = ", self.iteration)
            add_scalars(alg_tb_dict, self.writer, step=self.iteration)
            add_scalars(sampler_tb_dict, self.writer, step=self.iteration)

        # evaluate
        if self.iteration % self.eval_interval == 0 and self.use_eval:
            print("=================eval================")
            with ModuleOnDevice(self.networks, "cpu"):
                total_avg_return = self.evaluator.run_evaluation(self.iteration,self.sampler.env)

            if (
                total_avg_return >= self.best_tar
                and self.iteration >= self.max_iteration / 5
            ):
                self.best_tar = total_avg_return
                print("Best return = {}!".format(str(self.best_tar)))

                for filename in os.listdir(self.save_folder + "/apprfunc/"):
                    if filename.endswith("_opt.pkl"):
                        os.remove(self.save_folder + "/apprfunc/" + filename)

                torch.save(
                    self.networks.state_dict(),
                    self.save_folder
                    + "/apprfunc/apprfunc_{}_opt.pkl".format(self.iteration),
                )

            self.writer.add_scalar(
                tb_tags["Buffer RAM of RL iteration"],
                self.buffer.__get_RAM__(),
                self.iteration,
            )
            self.writer.add_scalar(
                tb_tags["TAR of RL iteration"], total_avg_return, self.iteration
            )
            self.writer.add_scalar(
                tb_tags["TAR of replay samples"],
                total_avg_return,
                self.iteration * self.replay_batch_size,
            )
            self.writer.add_scalar(
                tb_tags["TAR of total time"],
                total_avg_return,
                int(time.time() - self.start_time),
            )
            self.writer.add_scalar(
                tb_tags["TAR of collected samples"],
                total_avg_return,
                self.sampler.get_total_sample_number(),
            )

        # save
        if self.iteration % self.apprfunc_save_interval == 0:
            self.save_apprfunc()

    def train(self):
        self.buffer.save(self.save_folder)
        while self.iteration < self.max_iteration:
            self.step()
            self.iteration += 1

        self.save_apprfunc()
        self.writer.flush()

    def save_apprfunc(self):
        save_data = {
            "state_dict": self.networks.state_dict(),  # Save all model parameters
            "activate_rl": self.networks.activate_rl,
            "activate_rl_step_bound": self.networks.activate_rl_step_bound,
            "mean_std1_behavior": self.networks.mean_std1_behavior,
            "mean_std2_behavior": self.networks.mean_std2_behavior,
            "mean_std1_novice": self.networks.mean_std1_novice,
            "mean_std2_novice": self.networks.mean_std2_novice,

            "mean_std1_breward": self.networks.mean_std1_breward,
            "mean_std2_breward": self.networks.mean_std2_breward,
        }
        torch.save(
            save_data,
            self.save_folder + "/apprfunc/apprfunc_{}.pkl".format(self.iteration),
        )

    def load_apprfunc(self, file_path):
        checkpoint = torch.load(file_path)
        self.networks.load_state_dict(checkpoint["state_dict"])  # Load model parameters
        # Restore additional attributes
        self.networks.activate_rl = checkpoint["activate_rl"]
        self.networks.mean_std1_behavior = checkpoint["mean_std1_behavior"]
        self.networks.mean_std2_behavior = checkpoint["mean_std2_behavior"]
        self.networks.mean_std1_novice = checkpoint["mean_std1_novice"]
        self.networks.mean_std2_novice = checkpoint["mean_std2_novice"]

        self.networks.mean_std1_breward = checkpoint["mean_std1_breward"]
        self.networks.mean_std2_breward = checkpoint["mean_std2_breward"]
