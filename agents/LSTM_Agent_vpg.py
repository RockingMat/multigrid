import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import wandb
from agents.Agents import BaseAgent

class Agent_LSTM_VPG(BaseAgent):
    def __init__(self, config):
        super(Agent_LSTM_VPG, self).__init__(config)
        self.fc = nn.Sequential(
            nn.Linear(config.input_dim, 128),
            nn.ReLU(),
        )
        self.lstm_hidden_size = config.lstm_hidden_size
        self.lstm = nn.LSTM(input_size=128, hidden_size=self.lstm_hidden_size, 
                            num_layers=1, batch_first=True)
        self.actor_head = nn.Linear(self.lstm_hidden_size, config.action_dim)
        self.critic_head = nn.Linear(self.lstm_hidden_size, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        self.global_step = 0
        self.episode_hidden_states = []

    def reset_hidden_state(self, batch_size):
        return

    def forward(self, image, direction, hidden_state=None):
        batch_size = image.shape[0]
        image_flat = image.view(batch_size, -1)
        if direction.dim() == 2:
            direction = direction.squeeze(-1)
        direction_onehot = F.one_hot(direction.long(), num_classes=4).float()
        x = torch.cat([image_flat, direction_onehot], dim=1)
        x = self.fc(x)
        x = x.unsqueeze(1)
        hidden_state = (
            torch.zeros(1, batch_size, self.lstm_hidden_size),
            torch.zeros(1, batch_size, self.lstm_hidden_size)
        )
        lstm_out, new_hidden_state = self.lstm(x, hidden_state)
        lstm_out = lstm_out.squeeze(1)
        logits = self.actor_head(lstm_out)
        value = self.critic_head(lstm_out)
        return logits, value, new_hidden_state
    
    def get_action_and_value(self, image, direction):
        logits, value, hidden = self.forward(image, direction)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


    # def get_action_and_value(self, image, direction):
    #     batch_size = image.shape[0]
    #     if self.current_hidden_state is None:
    #         self.reset_hidden_state(batch_size)
    #     logits, value, new_hidden_state = self.forward(image, direction, self.current_hidden_state)
    #     self.current_hidden_state = new_hidden_state
    #     # Detach hidden state for TBPTT storage.
    #     h_detach = new_hidden_state[0].detach().clone()
    #     c_detach = new_hidden_state[1].detach().clone()
    #     self.episode_hidden_states.append((h_detach, c_detach))
        
    #     dist = Categorical(logits=logits)
    #     action = dist.sample()
    #     log_prob = dist.log_prob(action)
    #     return action, log_prob, value
    
    # def forward_sequence(self, obs_seq, dir_seq, hidden_state):
    #     """
    #     Processes an entire sequence (episode) in one go.
    #     obs_seq: tensor of shape (T, *obs_shape)
    #     dir_seq: tensor of shape (T,) containing the direction indices
    #     hidden_state: the (h, c) tuple to start the sequence
    #     Returns:
    #       logits: tensor of shape (T, action_dim)
    #       value: tensor of shape (T, 1)
    #       new_hidden_state: updated (h, c) tuple after processing the sequence
    #     """
    #     T = obs_seq.shape[0]
    #     obs_flat = obs_seq.view(T, -1)
    #     direction_onehot = F.one_hot(dir_seq.long(), num_classes=4).float()
    #     x = torch.cat([obs_flat, direction_onehot], dim=1)
    #     x = self.input_fc(x)
    #     x = x.unsqueeze(0)  # Add batch dimension (batch_size = 1)
    #     lstm_out, new_hidden_state = self.lstm(x, hidden_state)
    #     lstm_out = lstm_out.squeeze(0)
    #     logits = self.actor_head(lstm_out)
    #     value = self.critic_head(lstm_out)
    #     return logits, value, new_hidden_state

    def train_model(self, ep):
        for epoch in range(self.config.num_epochs):
            e = 0
            for episode in self.episode_data:
                e += 1
                # if epoch == 0 and e == 1:
                #     print("episode actions: ", episode["actions"])
                obs_seq = torch.stack(episode["obs_images"])  # shape (T, *obs_shape)
                dir_seq = torch.stack(episode["obs_direction"])  # shape (T,)
                actions_seq = torch.stack(episode["actions"]).long()  # shape (T,)
                rewards_seq = torch.tensor(episode["rewards"], dtype=torch.float32)  # shape (T,)
                # values_seq = torch.stack(episode["values"]).squeeze()  # shape (T,)

                # Compute discounted returns
                T = rewards_seq.shape[0]
                returns = torch.zeros_like(rewards_seq)
                R = 0.0
                for t in reversed(range(T)):
                    R = rewards_seq[t] + self.config.gamma * R
                    returns[t] = R

                advantages = returns # - values_seq.detach()
                # if epoch == 0 and e == 1:
                #     print("advantages: ", advantages)
                #     print("returns: ", returns)
                #     print("values_seq: ", values_seq)
                
                # Use the initial hidden state of the episode
                # h0, c0 = episode["hidden_states"][0]
                # hidden = (h0, c0)
                logits_seq = []
                for obs, dir in zip(obs_seq, dir_seq):
                    obs = obs.unsqueeze(0)
                    dir = dir.unsqueeze(0)
                    logits, _, _ = self.forward(obs, dir)
                    logits_seq.append(logits)

                # logits_seq, values_seq_pred, new_hidden_state = self.forward(obs_seq, dir_seq)
                logits_seq = torch.stack(logits_seq, dim=0)
                # Compute losses over the entire sequence
                dist_seq = Categorical(logits=logits_seq)
                log_probs_seq = dist_seq.log_prob(actions_seq)
                entropy = dist_seq.entropy().mean()
                policy_loss = - (log_probs_seq * advantages).mean() - self.config.ent_coef * entropy

                # values_seq_pred has shape (T, 1) and returns has shape (T,)
                # value_loss = F.mse_loss(values_seq_pred.squeeze(), returns)
                loss = policy_loss# + self.config.vf_coef * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                wandb.log({
                    "train/epoch": epoch,
                    "train/policy_loss": policy_loss.item(),
                    "train/total_loss": loss.item(),
                    "train/advantage_mean": advantages.mean().item(),
                    "train/return_mean": returns.mean().item(),
                }, step=self.global_step)
                
                self.global_step += 1

        self.episode_data = []
        self.episode_hidden_states = []


    # def train_model(self, ep):
    #     for epoch in range(self.config.num_epochs):
    #         e = 0
    #         for episode in self.episode_data:
    #             e += 1
    #             if epoch == 0 and e == 1:
    #                 print("episode actions: ", episode["actions"])
    #             obs_seq = torch.stack(episode["obs_images"])
    #             dir_seq = torch.stack(episode["obs_direction"])
    #             actions_seq = torch.stack(episode["actions"]).long()
    #             rewards_seq = torch.tensor(episode["rewards"], dtype=torch.float32)
    #             values_seq = torch.stack(episode["values"]).squeeze()

    #             T = rewards_seq.shape[0]
    #             returns = torch.zeros_like(rewards_seq)
    #             R = 0.0
    #             for t in reversed(range(T)):
    #                 R = rewards_seq[t] + self.config.gamma * R
    #                 returns[t] = R

    #             advantages = returns - values_seq.detach()
    #             if epoch == 0 and e == 1:
    #                 print("advantages: ", advantages)
    #                 print("returns: ", returns)
    #                 print ("values_seq: ", values_seq)
    #             k2 = self.config.k2
    #             t = 0
    #             while t < T:
    #                 chunk_end = min(t + k2, T)
    #                 chunk_obs = obs_seq[t:chunk_end]
    #                 chunk_dir = dir_seq[t:chunk_end]
    #                 h_t, c_t = episode["hidden_states"][t]
    #                 hidden = (h_t, c_t)
                    
    #                 logits_chunk, values_chunk, hidden = self.forward_chunk(chunk_obs, chunk_dir, hidden)
    #                 # if epoch == 0 and e==1 and t == 0:
    #                 #     print("logits_chunk: ", logits_chunk)
    #                 actions_chunk = actions_seq[t:chunk_end]
    #                 dist_chunk = Categorical(logits=logits_chunk)
    #                 log_probs_chunk = dist_chunk.log_prob(actions_chunk)
    #                 adv_chunk = advantages[t:chunk_end]
    #                 ret_chunk = returns[t:chunk_end]
    #                 entropy = dist_chunk.entropy().mean()
    #                 policy_loss = - (log_probs_chunk * adv_chunk).mean() - self.config.ent_coef * entropy

    #                 value_loss = F.mse_loss(values_chunk, ret_chunk)
                    
                    
    #                 loss =  policy_loss + self.config.vf_coef * value_loss
                    
    #                 self.optimizer.zero_grad()
    #                 loss.backward()
    #                 total_norm = 0
    #                 for p in self.parameters():
    #                     if p.grad is not None:
    #                         param_norm = p.grad.data.norm(2)
    #                         total_norm += param_norm.item() ** 2
    #                 total_norm = total_norm ** 0.5
    #                 # print("Gradient norm:", total_norm)

    #                 torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)
    #                 self.optimizer.step()
                    
    #                 wandb.log({
    #                     "train/epoch": epoch,
    #                     "train/chunk_step": t,
    #                     "train/policy_loss": policy_loss.item(),
    #                     #"train/value_loss": value_loss.item(),
    #                     "train/total_loss": loss.item(),
    #                     "train/advantage_mean": adv_chunk.mean().item(),
    #                     "train/return_mean": ret_chunk.mean().item(),
    #                 }, step=self.global_step)
                    
    #                 self.global_step += 1
    #                 hidden = (hidden[0].detach(), hidden[1].detach())
    #                 t = chunk_end

    #     self.episode_data = []
    #     self.episode_hidden_states = []