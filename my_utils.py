import torch
import numpy as np
from torch.distributions import Categorical
import torch.nn.functional as F

def compute_gae(rewards, values, gamma, lam):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        reward_t = rewards[t][0] if isinstance(rewards[t], list) else rewards[t]
        next_value = 0 if t == len(rewards) - 1 else values[t + 1]
        delta = reward_t + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns

def train_model(model, ep):
    """
    Update the agent's model using PPO.

    Parameters:
    obs_tensor (Tensor): Batch of observation images.
    dir_tensor (Tensor): Batch of observation directions.
    actions_tensor (Tensor): Batch of taken actions.
    old_log_probs_tensor (Tensor): Batch of log probabilities computed earlier.
    returns_tensor (Tensor): Batch of computed returns.
    advantages_tensor (Tensor): Batch of computed advantages.
    old_values_tensor (Tensor): Batch of value predictions computed earlier.
    ep (int): Current episode number (for learning rate annealing).
    """
    obs_tensor, dir_tensor, actions_tensor, old_log_probs_tensor, returns_tensor, advantages_tensor, old_values_tensor = collect_episode_data(model)

    dataset_size = obs_tensor.shape[0]
    indices = np.arange(dataset_size)
    epsilon = model.config.epsilon

    if model.config.lr_annealing:
        frac = 1.0 - (ep - 1) / model.config.n_episodes
        lr = model.config.learning_rate * frac
        for param_group in model.optimizer.param_groups:
            param_group['lr'] = lr

    # PPO update loop.
    for epoch in range(model.config.num_epochs):
        np.random.shuffle(indices)
        for start in range(0, dataset_size, model.config.mini_batch_size):
            end = start + model.config.mini_batch_size
            batch_idx = indices[start:end]

            batch_obs = obs_tensor[batch_idx]
            batch_dir = dir_tensor[batch_idx]
            batch_actions = actions_tensor[batch_idx]
            batch_old_log_probs = old_log_probs_tensor[batch_idx]
            batch_advantages = advantages_tensor[batch_idx]
            batch_returns = returns_tensor[batch_idx]
            batch_old_values = old_values_tensor[batch_idx]

            new_logits, new_values = model.forward(batch_obs, batch_dir)
            new_dist = Categorical(logits=new_logits)
            new_log_probs = new_dist.log_prob(batch_actions.squeeze())

            # policy loss with clipping
            ratio = torch.exp(new_log_probs - batch_old_log_probs.squeeze())
            surr1 = ratio * batch_advantages.squeeze()
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * batch_advantages.squeeze()
            policy_loss = -torch.min(surr1, surr2).mean()

            # value loss with clipping.
            value_pred_clipped = batch_old_values + (new_values - batch_old_values).clamp(-epsilon, epsilon)
            value_loss_unclipped = (new_values - batch_returns).pow(2)
            value_loss_clipped = (value_pred_clipped - batch_returns).pow(2)
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()

            entropy_bonus = new_dist.entropy().mean()
            loss = policy_loss + model.config.vf_coef * value_loss - model.config.ent_coef * entropy_bonus

            model.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), model.config.max_grad_norm)
            model.optimizer.step()

def collect_episode_data(model):
    batch_obs = []
    batch_dir = []
    batch_actions = []
    batch_old_log_probs = []
    batch_returns = []
    batch_advantages = []
    batch_old_values = []

    for episode in model.episode_data:
        rewards = np.array(episode["rewards"])
        values = torch.stack(episode["values"])  # shape: (T,)
        # Compute GAE and returns for this episode.
        adv, ret = compute_gae(rewards, values.numpy(), model.config.gamma, model.config.gae_lambda)
        advantages = torch.tensor(adv, dtype=torch.float32).unsqueeze(-1)
        returns_tensor = torch.tensor(ret, dtype=torch.float32).unsqueeze(-1)

        obs_tensor = torch.stack(episode["obs_images"])  # shape: (T, ...)
        dir_tensor = torch.stack(episode["obs_direction"])
        actions_tensor = torch.stack(episode["actions"]).long()
        log_probs_tensor = torch.stack(episode["log_probs"]).float()
        old_values = torch.stack(episode["values"]).float().unsqueeze(1)

        batch_obs.append(obs_tensor)
        batch_dir.append(dir_tensor)
        batch_actions.append(actions_tensor)
        batch_old_log_probs.append(log_probs_tensor)
        batch_returns.append(returns_tensor)
        batch_advantages.append(advantages)
        batch_old_values.append(old_values)

    # Combine all episodes for agent i.
    obs_tensor = torch.cat(batch_obs, dim=0)
    dir_tensor = torch.cat(batch_dir, dim=0)
    actions_tensor = torch.cat(batch_actions, dim=0)
    old_log_probs_tensor = torch.cat(batch_old_log_probs, dim=0)
    returns_tensor = torch.cat(batch_returns, dim=0)
    advantages_tensor = torch.cat(batch_advantages, dim=0)
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
    old_values_tensor = torch.cat(batch_old_values, dim=0)
    return obs_tensor, dir_tensor, actions_tensor, old_log_probs_tensor, returns_tensor, advantages_tensor, old_values_tensor