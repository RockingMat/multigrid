import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from my_utils import train_model

NUM_DIRECTIONS = 4

class Agent_Complex(nn.Module):
    def __init__(self, config):
        super(Agent_Complex, self).__init__()
        self.config = config

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=config.kernel_size, stride=1, padding=1),  # -> (B, 64, 5, 5)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=config.kernel_size, stride=1, padding=1),  # -> (B, 64, 5, 5)
            nn.ReLU(),
            nn.Flatten(),  # (B, 64*5*5 = 1600)
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU()
        )

        self.direction_embed = nn.Embedding(NUM_DIRECTIONS, config.fc_direction)

        self.actor_head = nn.Linear(128 + config.fc_direction, config.action_dim)
        self.critic_head = nn.Linear(128 + config.fc_direction, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        
    def forward(self, image, direction):
        """
        Process the image and direction and return the policy logits and state value.
        The image is assumed to be in (B, H, W, C) format.
        """
        if direction.dim() > 1:
            direction = direction.squeeze(-1)
        # (B, H, W, C) -> (B, C, H, W)
        c = self.cnn(image.permute(0, 3, 1, 2).float())
        d = self.direction_embed(direction.long())
        x = torch.cat([c, d], dim=-1)
        logits = self.actor_head(x)
        value = self.critic_head(x)
        return logits, value

    def get_action_and_value(self, image, direction):
        """
        Returns:
          action: sampled action
          log_prob: log probability of the sampled action
          value: the critic’s estimate of the state value
        """
        logits, value = self.forward(image, direction)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def load_model(self, save_path='checkpoint.pth'):
        checkpoint = torch.load(save_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def save_model(self, save_path='checkpoint.pth'):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, save_path)
    
    def train_model(self, obs_tensor, dir_tensor, actions_tensor, old_log_probs_tensor, 
                    returns_tensor, advantages_tensor, old_values_tensor, ep):
        train_model(self, obs_tensor, dir_tensor, actions_tensor, old_log_probs_tensor, 
                    returns_tensor, advantages_tensor, old_values_tensor, ep)
