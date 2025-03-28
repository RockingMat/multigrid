import torch


def run_debug_episode(env, agent, device, max_steps=500, visualize=False):
    """
    Runs a single debug (evaluation) episode using the current policy.
    Optionally captures frames to visualize or turn into a video later.
    
    Returns:
      total_reward: float, sum of rewards over the episode
      length: int, number of steps before done
      frames: list of RGB arrays (if visualize=True)
    """
    frames = []
    state, _ = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())  # take a random action to start
    done = False
    total_reward = 0.0
    length = 0

    # Convert to torch Tensors
    obs_image = torch.tensor(state['image'], dtype=torch.float32, device=device)
    obs_dir   = torch.tensor(state['direction'], device=device)
    
    while not done and length < max_steps:
        # Optionally store a rendered frame
        if visualize:
            # For many MultiGrid envs, you can do:
            frame = env.render(mode='rgb_array')  
            frames.append(frame)

        with torch.no_grad():
            # Agent forward pass
            action, logprob, value = agent.get_action_and_value(
                obs_image,  # make batch dimension
                obs_dir
            )
        # Convert action back to numpy
        action_np = action.cpu().numpy()

        # Step in environment
        next_state, reward, done, info = env.step(action_np)
        total_reward += reward[0]  # single-agent assumption
        length += 1

        # Update obs
        obs_image = torch.tensor(next_state['image'], dtype=torch.float32, device=device)
        obs_dir   = torch.tensor(next_state['direction'], device=device)

    return total_reward, length, frames

def compute_gae(rewards, values, gamma, lam):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        # If rewards[t] is a list, extract the scalar value
        reward_t = rewards[t][0] if isinstance(rewards[t], list) else rewards[t]
        next_value = 0 if t == len(rewards) - 1 else values[t + 1]
        delta = reward_t + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns

def one_hot_encode_image(image):
    """
    image: (B, height, width, 3) integer tensor
           last dimension has (type, color, state)
    returns: (B, 21, height, width) float tensor (one-hot)
    """
    # Convert to long just in case
    image = image.long()

    # The first channel: type in [0..10]
    type_oh   = F.one_hot(image[..., 0], num_classes=11)  # (B, H, W, 11)
    # The second channel: color in [0..5]
    color_oh  = F.one_hot(image[..., 1], num_classes=6)   # (B, H, W, 6)
    # The third channel: state in [0..3]
    state_oh  = F.one_hot(image[..., 2], num_classes=4)   # (B, H, W, 4)

    # Concatenate along the last dimension => shape: (B, H, W, 21)
    one_hot = torch.cat([type_oh, color_oh, state_oh], dim=-1)
    
    # Now reorder to (B, 21, H, W) for Conv2d
    one_hot = one_hot.permute(0, 3, 1, 2).float()
    return one_hot
