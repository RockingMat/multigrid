from copy import deepcopy
import gym
from itertools import count
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import wandb

from utils import plot_single_frame, make_video, extract_mode_from_path
#from simple_agent import Agent_Simple as Agent
from agents.complex_agent import Agent_Complex as Agent
from my_utils import compute_gae

class MultiAgent():
    """This is a meta agent that creates and controls several sub agents."""

    def __init__(self, config, env, device, training=True, with_expert=None, debug=False):
        # Store configuration and basic parameters.
        self.config = config
        self.env = env
        self.device = device
        self.training = training
        self.with_expert = with_expert
        self.debug = debug
        self.n_agents = self.config.n_agents
        self.zero_reward_count = 0

        self.agent = Agent(config)
        

   
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.backends.cudnn.deterministic = self.config.torch_deterministic
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.config.learning_rate, eps=1e-5)

        # TRY NOT TO MODIFY: start the game
        self.total_steps = 0       
        self.episode_data = []
        

    def run_one_episode(self, env, episode, log=True, train=True, save_model=True, visualize=False):
        # Reset environment and get initial observation.
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminations, infos = env.step(action)
        done = False

        if visualize:
            viz_data = self.init_visualization_data(env, obs)

        # Local buffers for episode transitions.
        episode_obs_images = []
        episode_obs_direction = []
        episode_actions = []
        episode_logprobs = []
        episode_values = []
        episode_rewards = []
        episode_dones = []

        while not done:
            self.total_steps += 1

            # Prepare batched observation tensors for all agents.
            # Prepare batched observation tensors for all agents.
            obs_image_tensor = torch.tensor(obs["image"], device=self.device)
            obs_direction_tensor = torch.tensor(obs["direction"], device=self.device)
            episode_obs_images.append(obs_image_tensor.clone())
            episode_obs_direction.append(obs_direction_tensor.clone())
            action, log_prob, value = self.agent.get_action_and_value(obs_image_tensor.clone(), obs_direction_tensor.clone())

            # Save actions and other outputs if training.
            if train:
                episode_actions.append(action.detach().cpu())
                episode_logprobs.append(log_prob.detach().cpu())
                episode_values.append(value.detach().cpu())

            # Execute the batched action.
            actions_np = action.detach().cpu().numpy()
            obs, reward, terminations, infos = env.step(actions_np)
            episode_rewards.append(reward)
            episode_dones.append(terminations)

            if visualize:
                viz_data = self.add_visualization_data(viz_data, env, obs, actions_np, obs)

            # End the episode if all agents are done.
            done = np.all(terminations)

        # Save the entire episode's data for later model updates.
        self.episode_data.append({
            "obs_images": episode_obs_images,
            "obs_direction": episode_obs_direction,
            "actions": episode_actions,
            "log_probs": episode_logprobs,
            "values": episode_values,
            "rewards": episode_rewards,
            "dones": episode_dones,
        })

        # Logging and checkpointing.
        if log:
            self.log_one_episode(episode, len(episode_rewards), episode_rewards)
        self.print_terminal_output(episode, np.sum(episode_rewards))
        if save_model:
            self.save_model_checkpoints(episode)

        if visualize:
            viz_data['rewards'] = np.array(episode_rewards)
            return viz_data

    def save_model_checkpoints(self, episode):
        if episode % self.config.save_model_episode == 0:
            self.agent.save_model()

    def print_terminal_output(self, episode, total_reward):
        if episode % self.config.print_every == 0:
            print('Total steps: {} \t Episode: {} \t Total reward: {}'.format(
                self.total_steps, episode, total_reward))

    def init_visualization_data(self, env, state):
        viz_data = {
            'agents_partial_images': [],
            'actions': [],
            'full_images': [],
            'predicted_actions': None
            }
        viz_data['full_images'].append(env.render('rgb_array'))

        return viz_data

    def add_visualization_data(self, viz_data, env, state, actions, next_state):
        viz_data['actions'].append(actions)
        viz_data['agents_partial_images'].append(
            [env.get_obs_render(
                self.get_agent_state(state, i)['image']) for i in range(self.n_agents)])
        viz_data['full_images'].append(env.render('rgb_array'))
        return viz_data
        
    def update_models(self, ep):
        # If no data has been collected, do nothing.
        if len(self.episode_data) == 0:
            return

        # Prepare lists to accumulate batch data.
        batch_obs = []
        batch_dir = []
        batch_actions = []
        batch_old_log_probs = []
        batch_returns = []
        batch_advantages = []
        batch_old_values = []

        for episode in self.episode_data:
            rewards = np.array(episode["rewards"])
            values = torch.stack(episode["values"])  
            T, num_agents = rewards.shape

            advantages_all = []
            returns_all = []
            for agent in range(num_agents):
                adv, ret = compute_gae(rewards[:, agent], values[:, agent].numpy(),
                                    self.config.gamma, self.config.gae_lambda)
                advantages_all.append(adv)
                returns_all.append(ret)
            advantages = torch.tensor(np.stack(advantages_all, axis=1), dtype=torch.float32, device=self.device).unsqueeze(-1)
            returns_tensor = torch.tensor(np.stack(returns_all, axis=1), dtype=torch.float32, device=self.device).unsqueeze(-1)

            # Process and flatten observations.
            # Each entry in episode["obs_images"] is already a tensor of shape (num_agents, 5, 5, 3)
            obs = torch.stack(episode["obs_images"])  # shape: (T, num_agents, 5,5,3)
            obs = obs.view(-1, *obs.shape[2:])  # Flatten time and agent dimensions.
            batch_obs.append(obs)

            # Process and flatten directional information.
            directions = torch.stack(episode["obs_direction"])  # shape: (T, num_agents, 3)
            directions = directions.view(-1, *directions.shape[2:])
            batch_dir.append(directions)

            # Process actions (each stored tensor has shape (num_agents,)).
            actions = torch.stack(episode["actions"])  # shape: (T, num_agents)
            actions = actions.view(-1)
            batch_actions.append(actions.long())

            # Process log_probs similarly.
            log_probs = torch.stack(episode["log_probs"])  # shape: (T, num_agents)
            log_probs = log_probs.view(-1)
            batch_old_log_probs.append(log_probs.float())

            # Flatten advantages and returns.
            batch_returns.append(returns_tensor.view(-1, 1))
            batch_advantages.append(advantages.view(-1, 1))

            # Process old values (flattened from (T, num_agents)).
            old_values = torch.stack(episode["values"])
            old_values = old_values.view(-1)
            batch_old_values.append(old_values.float().unsqueeze(1))

        # Combine data from all episodes into one batch.
        obs_tensor = torch.cat(batch_obs, dim=0)
        dir_tensor = torch.cat(batch_dir, dim=0)
        actions_tensor = torch.cat(batch_actions, dim=0)
        old_log_probs_tensor = torch.cat(batch_old_log_probs, dim=0)
        returns_tensor = torch.cat(batch_returns, dim=0)
        advantages_tensor = torch.cat(batch_advantages, dim=0)
        # Normalize the advantages.
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        old_values_tensor = torch.cat(batch_old_values, dim=0)

        dataset_size = obs_tensor.shape[0]
        indices = np.arange(dataset_size)

        # Retrieve PPO hyperparameters.
        epsilon = self.config.epsilon
        num_epochs = self.config.num_epochs
        mini_batch_size = self.config.mini_batch_size
        value_clip = self.config.value_clip
        max_grad_norm = self.config.max_grad_norm

        # adjust the learning rate using annealing.
        if self.config.lr_annealing:
            frac = 1.0 - (ep - 1) / self.config.n_episodes
            lr = self.config.learning_rate * frac
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        # PPO update loop.
        for epoch in range(num_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, mini_batch_size):
                end = start + mini_batch_size
                batch_idx = indices[start:end]
                batch_obs_mb = obs_tensor[batch_idx]
                batch_dir_mb = dir_tensor[batch_idx]
                batch_actions_mb = actions_tensor[batch_idx]
                batch_old_log_probs_mb = old_log_probs_tensor[batch_idx]
                batch_advantages_mb = advantages_tensor[batch_idx]
                batch_returns_mb = returns_tensor[batch_idx]
                batch_old_values_mb = old_values_tensor[batch_idx]

                # Forward pass: get new logits and value estimates.
                new_logits, new_values = self.agent.forward(batch_obs_mb, batch_dir_mb)
                new_dist = Categorical(logits=new_logits)
                new_log_probs = new_dist.log_prob(batch_actions_mb.squeeze())

                # Policy loss with clipping.
                ratio = torch.exp(new_log_probs - batch_old_log_probs_mb.squeeze())
                surr1 = ratio * batch_advantages_mb.squeeze()
                surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * batch_advantages_mb.squeeze()
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with clipping.
                value_pred_clipped = batch_old_values_mb + (new_values - batch_old_values_mb).clamp(-value_clip, value_clip)
                value_loss_unclipped = (new_values - batch_returns_mb).pow(2)
                value_loss_clipped = (value_pred_clipped - batch_returns_mb).pow(2)
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # Entropy bonus for exploration.
                entropy_bonus = new_dist.entropy().mean()
                loss = policy_loss + self.config.vf_coef * value_loss - self.config.ent_coef * entropy_bonus

                # Backpropagation and gradient clipping.
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_grad_norm)
                self.optimizer.step()

        # Clear the episode data after updating the model.
        self.episode_data = []

    
    def train(self, env):
        for episode in range(self.config.n_episodes):
            if episode % self.config.visualize_every == 0 and not (self.debug and episode == 0):
                viz_data = self.run_one_episode(env, episode, visualize=True)
                self.visualize(env, self.config.mode + '_training_step' + str(episode), 
                               viz_data=viz_data)
            else:
                self.run_one_episode(env, episode)
            
            if episode % self.config.update_every == 0:
                    self.update_models(episode)

        env.close()
        return

    def visualize(self, env, mode, video_dir='videos', viz_data=None):
        if not viz_data:
            viz_data = self.run_one_episode(
                env, episode=0, log=False, train=False, save_model=False, visualize=True)
            env.close()

        video_path = os.path.join(*[video_dir, self.config.experiment_name, self.config.model_name])

        # Set up directory.
        if not os.path.exists(video_path):
            os.makedirs(video_path)

        # Get names of actions
        action_dict = {}
        for act in env.Actions:
            action_dict[act.value] = act.name

        traj_len = len(viz_data['rewards'])
        for t in range(traj_len):
            self.visualize_one_frame(t, viz_data, action_dict, video_path, self.config.model_name)
            #print('Frame {}/{}'.format(t, traj_len))

        make_video(video_path, mode + '_trajectory_video')

    def visualize_one_frame(self, t, viz_data, action_dict, video_path, model_name):
        plot_single_frame(t, 
                          viz_data['full_images'][t], 
                          viz_data['agents_partial_images'][t], 
                          viz_data['actions'][t], 
                          viz_data['rewards'], 
                          action_dict, 
                          video_path, 
                          self.config.model_name, 
                          predicted_actions=viz_data['predicted_actions'], 
                          all_actions=viz_data['actions'])

    def load_models(self, model_path=None):
        if model_path is not None:
            self.agent.load_model(save_path=model_path + '_agent')
        else:
            # Use agents' default model path
            self.agent.load_model()

    def get_agent_state(self, state, agent_idx):
        """Extract the state for a specific agent from the environment's state."""
        return {
            "image": state["image"][agent_idx],
            "direction": state["direction"][agent_idx]
        }
    
    def log_one_episode(self, episode, length, rewards):
        """Log episode data to wandb or console."""
        if self.training and not self.debug:
            wandb.log({
                "episode": episode,
                "length": length,
                "total_reward": np.sum(rewards),
                "average_reward": np.mean(rewards)
            })
        else:
            print(f"Episode {episode}: Length = {length}, Total Reward = {np.sum(rewards)}, Average Reward = {np.mean(rewards)}")

