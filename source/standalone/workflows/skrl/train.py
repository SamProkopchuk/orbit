# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# added
parser.add_argument("--timesteps", type=int, default=None, help="Maximum number of timesteps for trainer.")
parser.add_argument("--usewandb", action="store_true", default=False, help="Use wandb logging.")
parser.add_argument("--render_mode", type=str, default=None, help="Render mode for gym env. Usually `rgb_array` or `None`.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
from datetime import datetime

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.utils import set_seed
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model, shared_model

from omni.isaac.orbit.utils.dict import print_dict
from omni.isaac.orbit.utils.io import dump_pickle, dump_yaml

import omni.isaac.orbit_tasks  # noqa: F401
from omni.isaac.orbit_tasks.utils import load_cfg_from_registry, parse_env_cfg
from omni.isaac.orbit_tasks.utils.wrappers.skrl import SkrlSequentialLogTrainer, SkrlVecEnvWrapper, process_skrl_cfg


import wandb
import time

def init_wandb(args):
    # if args.wandb_id is not None:
    #     print(f"Resuming wandb run with id={args.wandb_id}.")
    #     run = wandb.init(id=args.wandb_id, resume=True)

    print(f'wandb enabled: {args.usewandb}')
    if args.usewandb:
        # initialize wandb
        task = args.task
        task_rename = {
            "Isaac-Lift-Cube-Camera-Franka-v0": "visual",
            "Isaac-Lift-Cube-Franka-v0": "state",
        }
        task = task_rename[task] if task in task_rename else task
        run = wandb.init(
            project="orbit",
            # config=dict(args),
            config=args,
            name=f"{task}-n{args.num_envs}-s{args.seed}",
        )
        visual = task == 'visual'
        wandb.config.update({'visual': visual})
        # also saving to summary makes it easier to plot in wandb
        wandb.run.summary["num_envs"] = args.num_envs
        wandb.run.summary["seed"] = args.seed
        wandb.run.summary["task"] = args.task
        wandb.run.summary["visual"] = visual
        wandb.run.summary["timesteps"] = args.timesteps
        # automatically saves highest values to summary
        # wandb.define_metric("system.gpu.0.memoryAllocatedBytes", summary="max")
        # wandb.define_metric("system.proc.memory.rssMB", summary="max")

def main():
    """Train with skrl agent."""
            
    init_wandb(args_cli)
    start_setup_time = time.time()

    # read the seed from command line
    args_cli_seed = args_cli.seed
    
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO]: Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if experiment_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{experiment_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    experiment_cfg["agent"]["experiment"]["directory"] = log_root_path
    experiment_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), experiment_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), experiment_cfg)

    # create isaac environment
    render_mode = "rgb_array" if args_cli.video else None
    if args_cli.render_mode is not None:
        render_mode = args_cli.render_mode 
    print(f"[INFO]: Using render mode: {render_mode}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    # TODO(@Sam): See if need to pass rgb_array to added camera in the scene.
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO]: Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env)  # same as: `wrap_env(env, wrapper="isaac-orbit")`

    # Seems to not do anything (?):
    # env.sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    # set seed for the experiment (override from command line)
    set_seed(args_cli_seed if args_cli_seed is not None else experiment_cfg["seed"])

    # instantiate models using skrl model instantiator utility
    # https://skrl.readthedocs.io/en/latest/modules/skrl.utils.model_instantiators.html
    models = {}
    # non-shared models
    if experiment_cfg["models"]["separate"]:
        models["policy"] = gaussian_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            **process_skrl_cfg(experiment_cfg["models"]["policy"]),
        )
        models["value"] = deterministic_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            **process_skrl_cfg(experiment_cfg["models"]["value"]),
        )
    # shared models
    else:
        models["policy"] = shared_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            structure=None,
            roles=["policy", "value"],
            parameters=[
                process_skrl_cfg(experiment_cfg["models"]["policy"]),
                process_skrl_cfg(experiment_cfg["models"]["value"]),
            ],
        )
        models["value"] = models["policy"]

    # instantiate a RandomMemory as rollout buffer (any memory can be used for this)
    # https://skrl.readthedocs.io/en/latest/modules/skrl.memories.random.html
    memory_size = experiment_cfg["agent"]["rollouts"]  # memory_size is the agent's number of rollouts
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

    # configure and instantiate PPO agent
    # https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html
    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    experiment_cfg["agent"]["rewards_shaper"] = None  # avoid 'dictionary changed size during iteration'
    agent_cfg.update(process_skrl_cfg(experiment_cfg["agent"]))

    agent_cfg["state_preprocessor_kwargs"].update({"size": env.observation_space, "device": env.device})
    agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": env.device})

    agent = PPO(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )

    # configure and instantiate a custom RL trainer for logging episode events
    # https://skrl.readthedocs.io/en/latest/modules/skrl.trainers.base_class.html
    trainer_cfg = experiment_cfg["trainer"]
    if args_cli.timesteps is not None:
        trainer_cfg["timesteps"] = args_cli.timesteps
    trainer = SkrlSequentialLogTrainer(cfg=trainer_cfg, env=env, agents=agent)

    print('[INFO]: Sensors in the scene:', env.scene.sensors)
    print('[INFO]: Starting training.')

    # train the agent
    start_train_time = time.time()
    trainer.train(usewandb=args_cli.usewandb)
    traintime = time.time() - start_train_time
    print(f"Training time: {traintime} seconds.")

    # close the simulator
    env.close()

    runid = None
    if wandb.run is not None:
        runid = wandb.run.id
        wandb.run.summary["traintime"] = traintime
        wandb.run.summary["totaltime"] = time.time() - start_setup_time
        # since summary values can't be plotted in wandb (bug), add to config
        wandb.config.update({"traintime": traintime, "totaltime": time.time() - start_setup_time})
        wandb.finish()

    # add the max logged system/memory metrics to the summary
    if runid is not None:
        api = wandb.Api()
        # run = api.run("username/project/run_id")
        run = api.run(f"orbit/{runid}")
        keys = ["system.gpu.0.memoryAllocatedBytes", "system.proc.memory.rssMB"]
        system_metrics = run.history(stream="system") # events, systemMetrics
        for key in keys:
            if key in system_metrics:
                # Due to a bug added summary values won't be plotable
                run.summary[f"max_{key}"] = system_metrics[key].max()
                run.config.update({f"max_{key}": system_metrics[key].max()})
                # wandb.log({f"max_{key}": system_metrics[key].max()})
        run.summary.update()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
