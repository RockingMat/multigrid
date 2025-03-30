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
        # Use a configurable hidden size (default 128 if not provided)
        self.lstm_hidden_size = config.lstm_hidden_size
        self.lstm = nn.LSTM(input_size=128, hidden_size=self.lstm_hidden_size, 
                            num_layers=1, batch_first=True)
        # Separate heads for actor (policy) and critic (value).
        self.actor_head = nn.Linear(self.lstm_hidden_size, config.action_dim)
        self.critic_head = nn.Linear(self.lstm_hidden_size, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        # For rollout, we store the current hidden state and a per-episode list.
        self.current_hidden_state = None  
        self.episode_hidden_states = []  # list of (h, c) for each step

    def reset_hidden_state(self, batch_size):
        """Reset the LSTM hidden state at the beginning of an episode."""
        self.current_hidden_state = (torch.zeros(1, batch_size, self.lstm_hidden_size),
                                     torch.zeros(1, batch_size, self.lstm_hidden_size))
        self.episode_hidden_states = []

    def forward(self, image, direction, hidden_state=None):
        """
        Forward pass for one time step.
        Expects image of shape (batch, H, W, C) or (batch, -1) if already flattened,
        and direction of shape (batch,) or (batch, 1).
        Returns logits, value, and the new hidden state.
        """
        batch_size = image.shape[0]
        # Flatten the image.
        image_flat = image.view(batch_size, -1)
        if direction.dim() == 2:
            direction = direction.squeeze(-1)
        # One-hot encode the direction (assumes 4 possible directions).
        direction_onehot = F.one_hot(direction.long(), num_classes=4).float()
        # Concatenate features.
        x = torch.cat([image_flat, direction_onehot], dim=1)
        x = self.input_fc(x)  # (batch, 128)
        # Add a time dimension (length = 1) for the LSTM.
        x = x.unsqueeze(1)  # (batch, 1, 128)
        # If no hidden state is provided, initialize to zeros.
        if hidden_state is None:
            hidden_state = (torch.zeros(1, batch_size, self.lstm_hidden_size),
                            torch.zeros(1, batch_size, self.lstm_hidden_size))
        lstm_out, new_hidden_state = self.lstm(x, hidden_state)
        # Remove time dimension.
        lstm_out = lstm_out.squeeze(1)  # (batch, lstm_hidden_size)
        logits = self.actor_head(lstm_out)
        value = self.critic_head(lstm_out)
        return logits, value, new_hidden_state

    def get_action_and_value(self, image, direction):
        """
        Uses the recurrent forward pass.
        If self.config.with_lstm is True, the agent resets/updates its hidden state and
        also appends a detached copy to its internal buffer (for TBPTT during training).
        """
        batch_size = image.shape[0]
        # On the first step of an episode, reset hidden state.
        if self.current_hidden_state is None:
            self.reset_hidden_state(batch_size)
        logits, value, new_hidden_state = self.forward(image, direction, self.current_hidden_state)
        # Update internal hidden state.
        self.current_hidden_state = new_hidden_state
        # Store a detached copy of the hidden state for later TBPTT.
        h_detach = new_hidden_state[0].detach().clone()
        c_detach = new_hidden_state[1].detach().clone()
        self.episode_hidden_states.append((h_detach, c_detach))
        
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def forward_step(self, image, direction, hidden_state):
        """
        Helper function to process a single time step given an external hidden state.
        Used during TBPTT training.
        """
        return self.forward(image, direction, hidden_state)

    def train_model(self, ep):
        """
        Modified TBPTT training loop.
        Unroll for k2 steps (a chunk), compute the loss for the entire chunk,
        backpropagate, then detach the hidden state before the next chunk.
        """
        # Ensure that each episode has stored hidden states.
        for episode in self.episode_data:
            if "hidden_states" not in episode:
                # Use the agent’s per-episode hidden state buffer.
                episode["hidden_states"] = self.episode_hidden_states

        # Perform multiple PPO update epochs over the collected episodes.
        for epoch in range(self.config.num_epochs):
            # For each episode in the buffer:
            for episode in self.episode_data:
                # Convert lists to tensors.
                obs_seq = torch.stack(episode["obs_images"])       # shape: (T, ...)
                dir_seq = torch.stack(episode["obs_direction"])      # shape: (T, ...) 
                actions_seq = torch.stack(episode["actions"]).long()   # shape: (T,)
                old_log_probs_seq = torch.stack(episode["log_probs"]).squeeze()  # shape: (T,)
                rewards_seq = torch.tensor(episode["rewards"], dtype=torch.float32)
                values_seq = torch.stack(episode["values"]).squeeze()
                
                # Compute advantages and returns.
                adv_list, ret_list = compute_gae(rewards_seq.numpy(),
                                                values_seq.numpy(),
                                                self.config.gamma,
                                                self.config.gae_lambda)
                adv_seq = torch.tensor(adv_list, dtype=torch.float32)
                ret_seq = torch.tensor(ret_list, dtype=torch.float32)
                old_values_seq = torch.stack(episode["values"]).unsqueeze(1)  # shape: (T, 1)
                T = obs_seq.shape[0]
                
                # TBPTT: chunk length k2.
                k2 = self.config.k2  # chunk length for unrolling
                
                # Initialize the hidden state for the episode.
                if "hidden_states" in episode and len(episode["hidden_states"]) > 0:
                    h0, c0 = episode["hidden_states"][0]
                    hidden = (h0, c0)
                else:
                    hidden = (torch.zeros(1, 1, self.lstm_hidden_size),
                            torch.zeros(1, 1, self.lstm_hidden_size))
                
                t = 0
                while t < T:
                    chunk_end = min(t + k2, T)
                    chunk_loss = 0
                    # Process the current chunk (from t to chunk_end) without intermediate detachment.
                    for i in range(t, chunk_end):
                        obs_t = obs_seq[i].unsqueeze(0)  # add batch dimension
                        dir_t = dir_seq[i].unsqueeze(0)
                        logits, value, hidden = self.forward_step(obs_t, dir_t, hidden)
                        dist = Categorical(logits=logits)
                        log_prob = dist.log_prob(actions_seq[i])
                        
                        # PPO surrogate loss.
                        ratio = torch.exp(log_prob - old_log_probs_seq[i])
                        surr1 = ratio * adv_seq[i]
                        surr2 = torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon) * adv_seq[i]
                        policy_loss = -torch.min(surr1, surr2)
                        
                        # Value loss with clipping.
                        value = value.squeeze()
                        old_val = old_values_seq[i].squeeze()
                        value_pred_clipped = old_val + (value - old_val).clamp(-self.config.epsilon, self.config.epsilon)
                        value_loss = torch.max((value - ret_seq[i]).pow(2),
                                            (value_pred_clipped - ret_seq[i]).pow(2))
                        
                        # Entropy bonus.
                        entropy_bonus = dist.entropy()
                        loss = policy_loss + self.config.vf_coef * value_loss - self.config.ent_coef * entropy_bonus
                        chunk_loss += loss

                    # Average the loss over the chunk.
                    chunk_loss = chunk_loss / (chunk_end - t)
                    self.optimizer.zero_grad()
                    chunk_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    
                    # Detach the hidden state at the chunk boundary.
                    hidden = (hidden[0].detach(), hidden[1].detach())
                    t = chunk_end

        # Clear episode data and hidden state buffer after training.
        self.episode_data = []
        self.episode_hidden_states = []
