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
from agents.Agents import Agent_Simple as Agent
# from agents.complex_agent import Agent_Complex as Agent

class MultiAgent():
    """This is a meta agent that creates and controls several sub agents."""

    def __init__(self, config, env, device, training=True, with_expert=None, debug=False):
        self.config = config
        self.env = env
        self.device = device
        self.training = training
        self.with_expert = with_expert
        self.debug = debug
        self.n_agents = self.config.n_agents
        self.total_steps = 0

        self.agents = [Agent(config) for _ in range(self.n_agents)]
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.backends.cudnn.deterministic = self.config.torch_deterministic

    def run_one_episode(self, env, episode, log=True, train=True, save_model=True, visualize=False):
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminations, infos = env.step(action)
        done = False

        if visualize:
            viz_data = self.init_visualization_data(env, obs)

        # Create separate episode buffers for each agent.
        episode_buffers = [{
            "obs_images": [],
            "obs_direction": [],
            "actions": [],
            "log_probs": [],
            "values": [],
            "rewards": [],
            "dones": []
        } for _ in range(self.n_agents)]
        episode_rewards = []
        if self.config.with_rnd:
            intrinsic_reward_total = 0

        while not done:
            self.total_steps += 1
            actions = []
            # For each agent, get its own observation and compute action, log_prob, and value.
            for i in range(self.n_agents):
                # Convert each agent's observation to a tensor.
                obs_image_tensor = torch.tensor(obs["image"][i], device=self.device)
                obs_direction_tensor = torch.tensor(obs["direction"][i], device=self.device)
                # Save the current observation.
                episode_buffers[i]["obs_images"].append(obs_image_tensor.clone())
                episode_buffers[i]["obs_direction"].append(obs_direction_tensor.clone())

                # Each agent makes its decision independently.
                action, log_prob, value = self.agents[i].get_action_and_value(obs_image_tensor.unsqueeze(0), obs_direction_tensor.unsqueeze(0))
                if train:
                    episode_buffers[i]["actions"].append(action.detach().cpu())
                    episode_buffers[i]["log_probs"].append(log_prob.detach().cpu())
                    episode_buffers[i]["values"].append(value.detach().cpu())
                actions.append(action.detach().cpu().numpy())

            # Execute actions for all agents at once.
            obs, reward, terminations, infos = env.step(actions)
            episode_rewards.append(reward)
            for i in range(self.n_agents):
                if self.config.with_rnd:
                    intrinsic_reward = self.agents[i].rnd.get_intrinsic_reward(torch.tensor(obs["image"][i]), torch.tensor(obs["direction"][i])) * self.config.intrinsic_reward_coef
                    intrinsic_reward_total += intrinsic_reward
                    episode_buffers[i]["rewards"].append(reward[i] + intrinsic_reward)
                    if self.total_steps % self.config.rnd_update_every == 0:
                        self.agents[i].rnd.update()
                else:
                    episode_buffers[i]["rewards"].append(reward[i])
                episode_buffers[i]["dones"].append(terminations)

            if visualize:
                viz_data = self.add_visualization_data(viz_data, env, obs, actions, obs)
            done = terminations

        for i in range(self.n_agents):
            self.agents[i].episode_data.append(episode_buffers[i])
        
        # Logging and checkpointing.
        if log:
            if  self.config.with_rnd:
                self.log_one_episode(episode, len(episode_rewards), episode_rewards, intrinsic_reward_total)
            else:
                self.log_one_episode(episode, len(episode_rewards), episode_rewards)
        self.print_terminal_output(episode, np.sum(episode_rewards))
        if save_model:
            # Optionally save each agent's model.
            if episode % self.config.save_model_episode == 0:
                for agent in self.agents:
                    agent.save_model()
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
            'full_images': []
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
        for i in range(self.n_agents):
            if len(self.agents[i].episode_data) == 0:
                continue
            self.agents[i].train_model(ep)
            self.agents[i].episode_data = []

    
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
                          self.config.model_name)

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
    
    def log_one_episode(self, episode, length, rewards, intrinsic_reward=None):
        """Log episode data to wandb or console."""
        if self.training and not self.debug:
            if self.config.with_rnd:
                wandb.log({
                    "episode": episode,
                    "length": length,
                    "total_reward": np.sum(rewards),
                    "average_reward": np.mean(rewards),
                    "intrinsic_reward": intrinsic_reward
                })
            else:
                wandb.log({
                    "episode": episode,
                    "length": length,
                    "total_reward": np.sum(rewards),
                    "average_reward": np.mean(rewards)
                })
        else:
            if self.config.with_rnd:
                print(f"Episode {episode}: Length = {length}, Total Reward = {np.sum(rewards)}, Average Reward = {np.mean(rewards)}, Intrinsic Reward = {intrinsic_reward}")
            else:
                print(f"Episode {episode}: Length = {length}, Total Reward = {np.sum(rewards)}, Average Reward = {np.mean(rewards)}")

