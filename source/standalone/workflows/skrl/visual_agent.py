"""
https://github.com/Toni-SM/skrl/blob/main/docs/source/examples/isaacsim/torch_isaacsim_jetbot_ppo.py
"""

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# define models (stochastic and deterministic models) using mixins
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(nn.Conv2d(3, 32, kernel_size=8, stride=4),
                                 nn.ReLU(),
                                 nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                 nn.ReLU(),
                                 nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                 nn.ReLU(),
                                 nn.Flatten(),
                                 nn.Linear(9216, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 16),
                                 nn.Tanh(),
                                 nn.Linear(16, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, 32),
                                 nn.Tanh(),
                                 nn.Linear(32, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # view (samples, width * height * channels) -> (samples, width, height, channels)
        # permute (samples, width, height, channels) -> (samples, channels, width, height)
        x = self.net(inputs["states"].view(-1, *self.observation_space.shape).permute(0, 3, 1, 2))
        return 10 * torch.tanh(x), self.log_std_parameter, {}   # JetBotEnv action_space is -10 to 10

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Conv2d(3, 32, kernel_size=8, stride=4),
                                 nn.ReLU(),
                                 nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                 nn.ReLU(),
                                 nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                 nn.ReLU(),
                                 nn.Flatten(),
                                 nn.Linear(9216, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 16),
                                 nn.Tanh(),
                                 nn.Linear(16, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, 32),
                                 nn.Tanh(),
                                 nn.Linear(32, 1))

    def compute(self, inputs, role):
        # view (samples, width * height * channels) -> (samples, width, height, channels)
        # permute (samples, width, height, channels) -> (samples, channels, width, height)
        return self.net(inputs["states"].view(-1, *self.observation_space.shape).permute(0, 3, 1, 2)), {}

