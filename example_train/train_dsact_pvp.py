import argparse
import os
#import numpy as np

from utils.initialization import create_alg,create_buffer,create_env,create_sampler,create_evaluator,create_trainer
# from training.evaluator import create_evaluator
# from training.off_sampler import create_sampler
# from training.trainer import create_trainer
from utils.init_args import init_args
from utils.tensorboard_setup import start_tensorboard, save_tb_to_csv

os.environ["OMP_NUM_THREADS"] = "4"


if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    ################################################
    # Key Parameters for users
    parser.add_argument("--env_id", type=str, default="gym_metadrivepvp", help="id of environment")
    #gym_pendulum can be replaced by other envs in the env_gym folder, such as gym_ant, gym_walker2d... but more complex envs need bigger "max iteration" setting. U can refer to "dsac_mlp_humanoid_offserial.py" to set up.
    parser.add_argument("--algorithm", type=str, default="DSAC_V2_PVP")
    #set algorithm default to DSAC_V2, but it can be replaced by DSAC_V1 if you want to use the old version of DSAC.

    parser.add_argument("--enable_cuda", default=True, help="Enable CUDA")
    parser.add_argument("--seed", default=None, help="Enable CUDA")
    ################################################
    # 1. Parameters for environment
    parser.add_argument("--reward_scale", type=float, default=1, help="reward scale factor")
    parser.add_argument("--action_type", type=str, default="continu", help="Options: continu/discret")
    parser.add_argument("--is_render", type=bool, default=True, help="Draw environment animation")
    parser.add_argument("--is_adversary", type=bool, default=False, help="Adversary training")

    ################################################
    # 2.1 Parameters of value approximate function
    parser.add_argument(
        "--value_func_name",
        type=str,
        default="ActionValueDistri",
        help="Options: StateValue/ActionValue/ActionValueDis/ActionValueDistri",
    )
    parser.add_argument("--value_func_type", type=str, default="MLP", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS")
    value_func_type = parser.parse_known_args()[0].value_func_type
    parser.add_argument("--value_hidden_sizes", type=list, default=[256,256,256])
    parser.add_argument(
        "--value_hidden_activation", type=str, default="gelu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument("--value_output_activation", type=str, default="linear", help="Options: linear/tanh")
    parser.add_argument("--value_min_log_std", type=int, default=-8)
    parser.add_argument("--value_max_log_std", type=int, default=8)

    # 2.2 Parameters of policy approximate function
    parser.add_argument(
        "--policy_func_name",
        type=str,
        default="StochaPolicy",
        help="Options: None/DetermPolicy/FiniteHorizonPolicy/StochaPolicy",
    )
    parser.add_argument(
        "--policy_func_type", type=str, default="MLP", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS"
    )
    parser.add_argument(
        "--policy_act_distribution",
        type=str,
        default="TanhGaussDistribution",
        help="Options: default/TanhGaussDistribution/GaussDistribution",
    )
    policy_func_type = parser.parse_known_args()[0].policy_func_type
    parser.add_argument("--policy_hidden_sizes", type=list, default=[256,256,256])
    parser.add_argument(
        "--policy_hidden_activation", type=str, default="gelu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument("--policy_output_activation", type=str, default="tanh", help="Options: linear/tanh")
    parser.add_argument("--policy_min_log_std", type=int, default=-20)
    parser.add_argument("--policy_max_log_std", type=int, default=3)

    ################################################
    # 3. Parameters for RL algorithm
    parser.add_argument("--value_learning_rate", type=float, default=0.0005)
    parser.add_argument("--policy_learning_rate", type=float, default=0.0005)
    parser.add_argument("--alpha_learning_rate", type=float, default=0.0003)
    # special parameter
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--auto_alpha", type=bool, default=True)
    parser.add_argument("--alpha", type=bool, default=0.2)
    parser.add_argument("--delay_update", type=int, default=2)
    parser.add_argument("--TD_bound", type=float, default=1)
    parser.add_argument("--bound", default=True)

    ################################################
    # 4. Parameters for trainer
    parser.add_argument(
        "--trainer",
        type=str,
        default="pvp_trainer",
        help="Options: on_serial_trainer, on_sync_trainer, off_serial_trainer, off_async_trainer",
    )
    # Maximum iteration number
    parser.add_argument(
        "--ini_network_dir",
        type=str,
        default=None
    )
    trainer_type = parser.parse_known_args()[0].trainer

    # 4.1. Parameters for off_serial_trainer
    parser.add_argument(
        "--buffer_name", type=str, default="pvp_buffer")
    parser.add_argument("--max_iteration", type=int, default=5e6)
    # Size of collected samples before training
    parser.add_argument("--buffer_warm_size", type=int, default=500)
    # Max size of reply buffer
    parser.add_argument("--buffer_max_size", type=int, default=2*50000)
    # Batch size of replay samples from buffer
    parser.add_argument("--replay_batch_size", type=int, default=256)
    # Period owwwwwwwwwwwwwwwwwwwwwwwwf sampling
    parser.add_argument("--sample_interval", type=int, default=10)

    ################################################
    # 5. Parameters for sampler
    parser.add_argument(
        "--sampler_name", type=str, default="pvp_sampler")
    # Batch size of sampler for buffer store
    parser.add_argument("--sample_batch_size", type=int, default=10)
    # Add noise to action for better exploration
    parser.add_argument("--noise_params", type=dict, default=None)

    ################################################
    # 6. Parameters for evaluator
    parser.add_argument("--use_eval", type=bool, default=False)
    parser.add_argument("--evaluator_name", type=str, default="Pvp_evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=2)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--eval_save", type=str, default=False, help="save evaluation data")

    ################################################
    # 7. Data savings
    #parser.add_argument("--save_folder", type=str, default= "/home/wangwenxuan/gops_idp/gops/results/DSAC2/humanoid_r_0.2_sb_20_si_1_2")
    parser.add_argument("--save_folder", type=str, default= None)
    # Save value/policy every N updates
    parser.add_argument("--apprfunc_save_interval", type=int, default=5000)

    # Save key info every N updates
    parser.add_argument("--log_save_interval", type=int, default=10)
    ################################################

    ################################################
    # 8. Parameters for EGPO
    parser.add_argument("--conservative_alpha", type=float, default= 7)
    parser.add_argument("--use_conservative_loss", type=bool, default= True)
    parser.add_argument("--use_takeover_cost", type=bool, default= True)
    parser.add_argument("--takeover_lambda", type=float, default= 0.0)
    parser.add_argument("--cost_limit", type=int, default= 3)
    ################################################

    ################################################
    # 8. Parameters for PVP
    parser.add_argument("--discard_reward", type=bool, default= True)
    parser.add_argument("--takeover_stop_td", type=bool, default= False)
    parser.add_argument("--cql_coefficient", type=float, default= 1.0)
    parser.add_argument("--save_buffer", type=bool, default= True)
    parser.add_argument("--save_buffer_fre", type=int, default= 1000)
    parser.add_argument("--load_buffer", type=bool, default= True)
    parser.add_argument("--load_human_buffer", type=bool, default= True)
    parser.add_argument("--buffer_path", type=str, default= "home/carla/LZQW/DSAC-T-EGPO-V2/results/DSAC_V2_PVP_RL_gym_metadrivepvp/240906-153853/buf_25000.pkl")
    parser.add_argument("--human_buffer_path", type=str, default= "home/carla/LZQW/DSAC-T-EGPO-V2/results/DSAC_V2_PVP_RL_gym_metadrivepvp/240906-153853/human_buf_25000.pkl")
    parser.add_argument("--warm_up", type=bool, default= True)
    parser.add_argument("--q_bound", type=float, default= 5)
    parser.add_argument("--warm_up_step", type=int, default= 30000)
    ################################################


    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**args)
    args = init_args(env, **args)

    #start_tensorboard(args["save_folder"])
    # Step 1: create algorithm and approximate function
    alg = create_alg(**args)
    # Step 2: create sampler in trainer
    sampler = create_sampler(**args)
    # Step 3: create buffer in trainer
    buffer = create_buffer(**args)
    # Step 4: create evaluator in trainer
    evaluator = create_evaluator(env,**args)
    # Step 5: create trainer
    trainer = create_trainer(alg, sampler, buffer, evaluator, **args)

    ################################################
    # Start training ... ...
    trainer.train()
    print("Training is finished!")

    ################################################
    # Plot and save training figures
    #plot_all(args["save_folder"])
    save_tb_to_csv(args["save_folder"])
    print("Plot & Save are finished!")
