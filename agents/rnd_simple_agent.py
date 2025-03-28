import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
from my_utils import train_model

    
class Agent_Simple_RND(nn.Module):
    def __init__(self, config):
        self.config = config
        super(Agent_Simple_RND, self).__init__()
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

    def get_action_and_value(self, image, direction):
        logits, value = self.forward(image, direction)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value
    
    def train_model(self, obs_tensor, dir_tensor, actions_tensor, old_log_probs_tensor, 
                    returns_tensor, advantages_tensor, old_values_tensor, ep):
        train_model(self, obs_tensor, dir_tensor, actions_tensor, old_log_probs_tensor, 
                    returns_tensor, advantages_tensor, old_values_tensor, ep)
    
    def load_model(self, save_path='checkpoint.pth'):
        checkpoint = torch.load(save_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def save_model(self, save_path='checkpoint.pth'):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, save_path)