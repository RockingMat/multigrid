import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from my_utils import compute_gae
from agents.Agents import BaseAgent

class Agent_LSTM(BaseAgent):
    def __init__(self, config):
        super(Agent_LSTM, self).__init__(config)
        self.input_fc = nn.Sequential(
            nn.Linear(config.input_dim, 128),
            nn.ReLU(),
        )
        self.lstm_hidden_size = config.lstm_hidden_size
        self.lstm = nn.LSTM(input_size=128, hidden_size=self.lstm_hidden_size, 
                            num_layers=1, batch_first=True)
        self.actor_head = nn.Linear(self.lstm_hidden_size, config.action_dim)
        self.critic_head = nn.Linear(self.lstm_hidden_size, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        self.current_hidden_state = None  
        self.episode_hidden_states = []  # list of (h, c) for each step

    def reset_hidden_state(self, batch_size):
        init_hidden = (
            torch.zeros(1, batch_size, self.lstm_hidden_size),
            torch.zeros(1, batch_size, self.lstm_hidden_size)
        )
        self.current_hidden_state = init_hidden
        self.episode_hidden_states = [init_hidden]

    def forward(self, image, direction, hidden_state=None):
        # This version is for single time-step (used during rollouts)
        batch_size = image.shape[0]
        image_flat = image.view(batch_size, -1)
        if direction.dim() == 2:
            direction = direction.squeeze(-1)
        direction_onehot = F.one_hot(direction.long(), num_classes=4).float()
        x = torch.cat([image_flat, direction_onehot], dim=1)
        x = self.input_fc(x)
        x = x.unsqueeze(1)
        if hidden_state is None:
            hidden_state = (
                torch.zeros(1, batch_size, self.lstm_hidden_size),
                torch.zeros(1, batch_size, self.lstm_hidden_size)
            )
        lstm_out, new_hidden_state = self.lstm(x, hidden_state)
        lstm_out = lstm_out.squeeze(1)
        logits = self.actor_head(lstm_out)
        value = self.critic_head(lstm_out)
        return logits, value, new_hidden_state

    def forward_chunk(self, obs_chunk, dir_chunk, hidden_state):
        """
        Processes a whole chunk (sequence) in one go.
        obs_chunk: tensor of shape (chunk_len, *obs_shape)
        dir_chunk: tensor of shape (chunk_len,) containing the direction indices
        hidden_state: the (h, c) tuple to start the chunk
        Returns:
          logits: tensor of shape (chunk_len, action_dim)
          value: tensor of shape (chunk_len, 1)
          new_hidden_state: updated (h, c) tuple after processing the chunk
        """
        chunk_len = obs_chunk.shape[0]
        obs_flat = obs_chunk.view(chunk_len, -1)
        direction_onehot = F.one_hot(dir_chunk.long(), num_classes=4).float()
        x = torch.cat([obs_flat, direction_onehot], dim=1)
        x = self.input_fc(x)
        x = x.unsqueeze(0)
        lstm_out, new_hidden_state = self.lstm(x, hidden_state)
        lstm_out = lstm_out.squeeze(0)
        logits = self.actor_head(lstm_out)
        value = self.critic_head(lstm_out)
        return logits, value, new_hidden_state

    def get_action_and_value(self, image, direction):
        batch_size = image.shape[0]
        if self.current_hidden_state is None:
            self.reset_hidden_state(batch_size)
        logits, value, new_hidden_state = self.forward(image, direction, self.current_hidden_state)
        self.current_hidden_state = new_hidden_state
        # Detach hidden state for TBPTT storage.
        h_detach = new_hidden_state[0].detach().clone()
        c_detach = new_hidden_state[1].detach().clone()
        self.episode_hidden_states.append((h_detach, c_detach))
        
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def train_model(self, ep):
        for epoch in range(self.config.num_epochs):
            for episode in self.episode_data:
                obs_seq = torch.stack(episode["obs_images"])       # shape: (T, *obs_shape)
                dir_seq = torch.stack(episode["obs_direction"])      # shape: (T,)
                actions_seq = torch.stack(episode["actions"]).long()   # shape: (T,)
                old_log_probs_seq = torch.stack(episode["log_probs"]).squeeze()  # shape: (T,)
                rewards_seq = torch.tensor(episode["rewards"], dtype=torch.float32)
                values_seq = torch.stack(episode["values"]).squeeze()  # shape: (T,) or (T, 1)
                
                # Compute advantages and returns.
                adv_list, ret_list = compute_gae(rewards_seq.numpy(),
                                                 values_seq.numpy(),
                                                 self.config.gamma,
                                                 self.config.gae_lambda)
                adv_seq = torch.tensor(adv_list, dtype=torch.float32)
                ret_seq = torch.tensor(ret_list, dtype=torch.float32)
                old_values_seq = torch.stack(episode["values"]).unsqueeze(1)  # shape: (T, 1)
                T = obs_seq.shape[0]
                k2 = self.config.k2  # chunk length

                h0, c0 = episode["hidden_states"][0]
                hidden = (h0, c0)

                t = 0
                while t < T:
                    # hidden = episode["hidden_states"][t]
                    chunk_end = min(t + k2, T)
                    chunk_obs = obs_seq[t:chunk_end]
                    chunk_dir = dir_seq[t:chunk_end]
                    logits_chunk, values_chunk, hidden = self.forward_chunk(chunk_obs, chunk_dir, hidden)
                    
                    dist_chunk = Categorical(logits=logits_chunk)
                    actions_chunk = actions_seq[t:chunk_end]
                    log_probs_chunk = dist_chunk.log_prob(actions_chunk)
                    old_log_probs_chunk = old_log_probs_seq[t:chunk_end]
                    adv_chunk = adv_seq[t:chunk_end]
                    
                    ratio = torch.exp(log_probs_chunk - old_log_probs_chunk)
                    surr1 = ratio * adv_chunk
                    surr2 = torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon) * adv_chunk
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    values_chunk = values_chunk.squeeze(1)
                    old_values_chunk = old_values_seq[t:chunk_end].squeeze()
                    ret_chunk = ret_seq[t:chunk_end]
                    value_pred_clipped = old_values_chunk + (values_chunk - old_values_chunk).clamp(-self.config.epsilon, self.config.epsilon)
                    value_loss = torch.max((values_chunk - ret_chunk).pow(2),
                                           (value_pred_clipped - ret_chunk).pow(2)).mean()
                    
                    entropy_bonus = dist_chunk.entropy().mean()
                    loss = policy_loss + self.config.vf_coef * value_loss - self.config.ent_coef * entropy_bonus

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    hidden = (hidden[0].detach(), hidden[1].detach())
                    t = chunk_end

        self.episode_data = []
        self.episode_hidden_states = []
