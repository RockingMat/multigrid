import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from abc import ABC, abstractmethod
from my_utils import train_model as Train
from RND_network import RND
import numpy as np
from my_utils import compute_gae

class BaseAgent(nn.Module, ABC):
    def __init__(self, config):
        super(BaseAgent, self).__init__()
        self.config = config
        self.episode_data = []
        if self.config.with_rnd:
            self.rnd = RND(config)
    
    @abstractmethod
    def forward(self, image, direction):
        """
        Process the input data (image and direction) and return:
            logits: action logits for the policy,
            value: the critic's estimate of the state value.
        """
        pass
    
    def get_action_and_value(self, image, direction):
        logits, value = self.forward(image, direction)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def train_model(self, ep):
        Train(self, ep)
    
    def load_model(self, save_path='checkpoint.pth'):
        checkpoint = torch.load(save_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def save_model(self, save_path='checkpoint.pth'):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, save_path)


class Agent_Simple(BaseAgent):
    def __init__(self, config):
        super(Agent_Simple, self).__init__(config)
        self.fc = nn.Sequential(
            nn.Linear(config.input_dim, 128),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, config.action_dim)
        self.value_head = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
    
    def forward(self, image, direction):
        batch_size = image.shape[0]
        image_flat = image.view(batch_size, -1)
        if direction.dim() == 2:
            direction = direction.squeeze(-1)
        direction_onehot = F.one_hot(direction.long(), num_classes=4).float()
        x = torch.cat([image_flat, direction_onehot], dim=1)
        x = self.fc(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value


class Agent_Complex(BaseAgent):
    def __init__(self, config):
        super(Agent_Complex, self).__init__(config)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=config.kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=config.kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU()
        )
        NUM_DIRECTIONS = 4
        self.direction_embed = nn.Embedding(NUM_DIRECTIONS, config.fc_direction)
        self.actor_head = nn.Linear(128 + config.fc_direction, config.action_dim)
        self.critic_head = nn.Linear(128 + config.fc_direction, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
    
    def forward(self, image, direction):
        if direction.dim() > 1:
            direction = direction.squeeze(-1)
        # Permute image from (B, H, W, C) to (B, C, H, W)
        c = self.cnn(image.permute(0, 3, 1, 2).float())
        d = self.direction_embed(direction.long())
        x = torch.cat([c, d], dim=-1)
        logits = self.actor_head(x)
        value = self.critic_head(x)
        return logits, value
