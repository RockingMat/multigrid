import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

# Define a small network used for both the target and predictor.
class RNDModel(nn.Module):
    def __init__(self, config):
        super(RNDModel, self).__init__()
        # Process the image with a simple CNN.
        # The image is 5x5 with 3 channels; we use one conv layer then flatten.
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),  # output: (32, 5, 5)
            nn.ReLU(),
            nn.Flatten()  # flatten to vector of size 32*5*5
        )
        # Fully connected layers that take the concatenated image features and direction.
        # We assume that directions are integers in [0, 3] so we use one-hot encoding (size 4).
        # The input dimension to the fc is: (32*5*5 + 4)
        self.fc = nn.Sequential(
            nn.Linear(32 * 5 * 5 + 4, config.rnd_hidden_size),
            nn.ReLU(),
            nn.Linear(config.rnd_hidden_size, config.rnd_output_size)
        )

    def forward(self, image, direction):
        # Expect image shape: (batch, 5, 5, 3) -> convert to (batch, 3, 5, 5)
        x = image.permute(0, 3, 1, 2).float()
        x = self.cnn(x)
        # Process the scalar direction by converting to one-hot vector (assume 4 possible directions)
        # If direction comes as (batch,1) we squeeze to (batch,)
        if direction.dim() > 1:
            direction = direction.squeeze(-1)
        direction_onehot = F.one_hot(direction.long(), num_classes=4).float()
        # Concatenate the flattened image features with the one-hot direction vector.
        x = torch.cat([x, direction_onehot], dim=1)
        return self.fc(x)


class RND(nn.Module):
    def __init__(self, config):
        """
        config must have the following attributes:
            - rnd_hidden_size: hidden dimension for the fc layers.
            - rnd_output_size: output dimension of the RND networks.
            - rnd_learning_rate: learning rate for training the predictor.
            - rnd_batch_size: batch size to sample from the internal buffer for updates.
        """
        super(RND, self).__init__()
        self.config = config
        
        self.target = RNDModel(config)
        self.predictor = RNDModel(config)
        
        for param in self.target.parameters():
            param.requires_grad = False
        
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=config.rnd_learning_rate)
        self.criterion = nn.MSELoss()
        
        self.buffer = []
    
    def forward(self, image, direction):
        """
        Given batched inputs, compute the outputs of the predictor and target.
        """
        pred = self.predictor(image, direction)
        with torch.no_grad():
            target_out = self.target(image, direction)
        return pred, target_out

    def get_intrinsic_reward(self, image, direction):
        """
        Given a single observation:
            - image: tensor or array of shape (5, 5, 3)
            - direction: scalar (integer)
        Computes and returns the intrinsic reward as the mean squared error between
        the predictor and target network outputs.
        """
        self.buffer.append((image, direction))
        image = image.unsqueeze(0)
        direction = direction.unsqueeze(0)
        pred, target_out = self.forward(image, direction)
        error = F.mse_loss(pred, target_out, reduction='none').mean(dim=1)
        return error.item()

    def update(self):
        """
        Given a batch of observations, perform one update step on the predictor network.
        """
        if len(self.buffer) < self.config.rnd_batch_size:
            return
        batch = random.sample(self.buffer, self.config.rnd_batch_size)
        self.buffer = []
        batch_images = torch.stack([item[0] for item in batch])  # shape: (B, 5, 5, 3)
        batch_directions = torch.stack([item[1] for item in batch])  # shape: (B,) or (B,1)
        pred, target_out = self.forward(batch_images, batch_directions)
        loss = self.criterion(pred, target_out.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
