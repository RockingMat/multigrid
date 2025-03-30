import unittest
from types import SimpleNamespace
import torch
from RND_network import RND  # Ensure your RND implementation is accessible

class TestRND(unittest.TestCase):
    def setUp(self):
        # Setup configuration and create the RND instance.
        config_dict = {
            'rnd_hidden_size': 128,
            'rnd_output_size': 64,
            'rnd_learning_rate': 0.001,
            'rnd_batch_size': 32,
        }
        self.config = SimpleNamespace(**config_dict)
        self.rnd = RND(self.config)
        # For get_intrinsic_reward (which unsqueezes the input), use an unbatched image.
        self.sample_image = torch.rand(5, 5, 3)   # shape: (5, 5, 3)
        self.sample_direction = torch.tensor(1)     # a scalar tensor (0-dim); unsqueeze will yield shape (1,)
        
        # For direct network calls (predictor/target), we need a batched input.
        self.sample_image_batched = self.sample_image.unsqueeze(0)  # shape: (1, 5, 5, 3)
        self.sample_direction_batched = self.sample_direction.unsqueeze(0)  # shape: (1,)

    def test_output_shapes_and_ranges(self):
        """
        Test that the predictor and target networks produce outputs of the same shape
        and that the predictor output is non-trivial.
        """
        with torch.no_grad():
            pred_output = self.rnd.predictor(self.sample_image_batched, self.sample_direction_batched)
            target_output = self.rnd.target(self.sample_image_batched, self.sample_direction_batched)
        # Check that both outputs have the same shape.
        self.assertEqual(
            pred_output.shape, target_output.shape, 
            "Predictor and target output shapes differ."
        )
        # Check that the predictor's output is non-trivial.
        self.assertFalse(
            torch.all(pred_output == 0), 
            "Predictor output is all zeros."
        )
        print("Predictor output range:", pred_output.min().item(), pred_output.max().item())
        print("Target output range:", target_output.min().item(), target_output.max().item())

    def test_intrinsic_reward_decreases(self):
        """
        For a fixed input, verify that the intrinsic reward decreases after multiple update steps.
        
        Steps:
          1. Compute an initial intrinsic reward.
          2. Fill the buffer with repeated samples.
          3. Run multiple update steps.
          4. Compare the intrinsic reward to ensure it decreases.
        """
        # Use the unbatched sample because get_intrinsic_reward will add a batch dimension.
        initial_reward = self.rnd.get_intrinsic_reward(self.sample_image, self.sample_direction)
        print("Initial intrinsic reward:", initial_reward)

        # Fill the buffer so that we have enough samples for an update.
        for _ in range(self.config.rnd_batch_size - 1):
            self.rnd.get_intrinsic_reward(self.sample_image, self.sample_direction)

        # Run several update steps.
        num_updates = 100
        for _ in range(num_updates):
            self.rnd.get_intrinsic_reward(self.sample_image, self.sample_direction)
            _ = self.rnd.update()

        # Compute the intrinsic reward after training.
        final_reward = self.rnd.get_intrinsic_reward(self.sample_image, self.sample_direction)
        print("Final intrinsic reward:", final_reward)
        self.assertLess(
            final_reward, initial_reward,
            "Intrinsic reward did not decrease after training."
        )

if __name__ == '__main__':
    unittest.main()
