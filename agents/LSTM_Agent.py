import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from abc import ABC, abstractmethod
from agents.Agents import BaseAgent
from my_utils import train_model_lstm as Train

class BaseLSTMAgent(BaseAgent, ABC):
    @abstractmethod
    def input_forward(self, image, direction):
        pass


    def train_model(self, ep):
        Train(self, ep)
    
    def get_action_and_value(self, image, direction, hidden_state):
        logits, value, new_hidden_state = self.forward(image, direction, hidden_state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, new_hidden_state

    def forward(self, image, direction, hidden_state):
        x = self.input_forward(image, direction)
        x = x.unsqueeze(1)
        lstm_out, new_hidden_state = self.lstm(x, hidden_state)
        lstm_out = lstm_out.squeeze(1)
        logits = self.actor_head(lstm_out)
        value = self.critic_head(lstm_out)
        return logits, value, new_hidden_state

    def episode_forward(self, obs_images, obs_direction):
        T = obs_images.shape[0]
        device = obs_images.device
        hidden_state = (torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size, device=device),
                        torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size, device=device))
        
        logits_seq = []
        values_seq = []
        for t in range(T):
            obs_t = obs_images[t].unsqueeze(0)
            dir_t = obs_direction[t].unsqueeze(0)
            logits, value, hidden_state = self.forward(obs_t, dir_t, hidden_state)
            logits_seq.append(logits.squeeze(0))
            values_seq.append(value.squeeze(0))
        
        logits_seq = torch.stack(logits_seq, dim=0)
        values_seq = torch.stack(values_seq, dim=0)
        return logits_seq, values_seq


class Agent_LSTM_Simple(BaseLSTMAgent):
    def __init__(self, config):
        super(Agent_LSTM_Simple, self).__init__(config)
        self.lstm_hidden_size = config.lstm_hidden_size
        self.input_fc = nn.Sequential(
            nn.Linear(config.input_dim, 128),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=config.lstm_hidden_size,
                            num_layers=1,
                            batch_first=True)
        self.actor_head = nn.Linear(config.lstm_hidden_size, config.action_dim)
        self.critic_head = nn.Linear(config.lstm_hidden_size, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)

    def input_forward(self, image, direction):
        batch_size = image.shape[0]
        image_flat = image.view(batch_size, -1)
        if direction.dim() == 2:
            direction = direction.squeeze(-1)
        direction_onehot = F.one_hot(direction.long(), num_classes=4).float()
        x = torch.cat([image_flat, direction_onehot], dim=1)
        x = self.input_fc(x)
        return x


class Agent_LSTM_Complex(BaseLSTMAgent):
    def __init__(self, config):
        super(Agent_LSTM_Complex, self).__init__(config)
        self.lstm_hidden_size = config.lstm_hidden_size
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=config.kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=config.kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU()
        )
        self.direction_embed = nn.Embedding(4, config.fc_direction)
        self.input_fc = nn.Linear(128 + config.fc_direction, 256)
        self.lstm = nn.LSTM(input_size=256,
                            hidden_size=config.lstm_hidden_size,
                            num_layers=1,
                            batch_first=True)
        self.actor_head = nn.Linear(config.lstm_hidden_size, config.action_dim)
        self.critic_head = nn.Linear(config.lstm_hidden_size, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
    
    def input_forward(self, image, direction):
        cnn_features = self.cnn(image.permute(0, 3, 1, 2).float())
        if direction.dim() > 1:
            direction = direction.squeeze(-1)
        dir_embedded = self.direction_embed(direction.long())
        x = torch.cat([cnn_features, dir_embedded], dim=-1)
        x = F.relu(self.input_fc(x))
        return x
