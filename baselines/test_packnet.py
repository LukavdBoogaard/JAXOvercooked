import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import unittest
from typing import List, Tuple, Dict, Any, NamedTuple

# Import your PackNet implementation
from baselines.IPPO_MLP_Packnet import Packnet, PacknetState

class TestPacknet(unittest.TestCase):
    """Tests for the PackNet implementation."""

    def setUp(self):
        """Set up a simple model and PackNet instance for testing."""
        # Define a simple network structure for testing
        self.seq_length = 2
        self.input_size = 10
        self.hidden_size = 20
        self.output_size = 5
        
        # Create deterministic parameters for testing
        key = jax.random.PRNGKey(42)
        
        # Create a simple two-layer network params dictionary
        params = {
            "params": {
                "Dense_0": {
                    "kernel": jax.random.normal(key, (self.input_size, self.hidden_size)),
                    "bias": jnp.zeros(self.hidden_size)
                },
                "Dense_1": {
                    "kernel": jax.random.normal(key, (self.hidden_size, self.output_size)),
                    "bias": jnp.zeros(self.output_size)
                }
            }
        }
        
        self.params = params["params"]
        self.packnet = Packnet(
            seq_length=self.seq_length,
            prune_instructions=0.5,  # Prune 50% of weights
            train_finetune_split=(100, 50),
            prunable_layers=[nn.Dense]
        )
        
        # Initialize PackNet state
        self.packnet_state = PacknetState(
            masks=self.packnet.init_mask_tree(self.params),
            current_task=0,
            train_mode=True
        )

    def test_mask_initialization(self):
        """Test that masks are initialized correctly."""
        masks = self.packnet_state.masks
        
        # Check mask structure matches params structure
        self.assertEqual(set(masks.keys()), set(self.params.keys()))
        
        # Each mask should be initialized to all False (nothing masked yet)
        for layer_name, layer_masks in masks.items():
            for param_name, mask in layer_masks.items():
                if "kernel" in param_name:
                    # The masks have the same shape as the parameters with a leading dimension for tasks
                    expected_shape = (self.seq_length,) + self.params[layer_name][param_name].shape
                    self.assertEqual(mask.shape, expected_shape)

                    # Check that all values in the mask are initialized as False
                    for task_idx in range(self.seq_length):
                        task_mask = mask[task_idx]
                        self.assertTrue(jnp.all(~task_mask), 
                                    f"Initial mask for task {task_idx} should have all values False")



    def test_prune_quantile_computation(self):
        """Test that prune quantile is correctly computed."""
        # Create specific params with known values for testing, including bias
        test_params = {
            "Dense_0": {
                "kernel": jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
                "bias": jnp.array([0.01, 0.02, 0.03])
            },
            "Dense_1": {
                "kernel": jnp.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]),
                "bias": jnp.array([0.04, 0.05])
            }
        }
        
        # Manually flatten all weights (only kernel params are prunable)
        all_weights = jnp.concatenate([
            test_params["Dense_0"]["kernel"].flatten(),
            test_params["Dense_1"]["kernel"].flatten()
        ])

    
        # Calculate expected cutoff at 50% quantile
        expected_cutoff = jnp.quantile(jnp.abs(all_weights), 0.5)
        print(f"Expected cutoff: {expected_cutoff}")
        
        # Create a packnet state with our test parameters
        test_packnet_state = PacknetState(
            masks=self.packnet.init_mask_tree(test_params),
            current_task=0,
            train_mode=True
        )
        
        # Prune the parameters 
        pruned_params, _ = self.packnet.prune(test_params, 0.5, test_packnet_state)
        
        # Verify weights below cutoff are zeroed and above cutoff are preserved
        for layer_name, layer_dict in pruned_params.items():
            for param_name, param in layer_dict.items():
                if "kernel" in param_name:
                    # Original param for comparison
                    orig_param = test_params[layer_name][param_name]
                    
                    # Check if values below cutoff are zeroed
                    should_be_pruned = jnp.abs(orig_param) < expected_cutoff
                    self.assertTrue(jnp.all(param[should_be_pruned] <= 0.00001),
                                f"Values below cutoff {expected_cutoff} should be zeroed")
                    
                    # Check if values above cutoff are preserved
                    should_be_kept = jnp.abs(orig_param) >= expected_cutoff
                    self.assertTrue(jnp.allclose(
                        param[should_be_kept],
                        orig_param[should_be_kept]
                    ), "Values above cutoff should be preserved")
                
                # Verify bias parameters are not pruned
                if "bias" in param_name:
                    self.assertTrue(jnp.allclose(
                        param,
                        test_params[layer_name][param_name]
                    ), "Bias parameters should not be pruned")



    def test_mask_combination(self):
        """Test that masks are correctly combined across tasks."""
        
        # Create a mask tree that mirrors the parameters
        mask_tree = {
            "Dense_0": {
                "bias": jnp.array([[False, True, True], [False, False, False]]),
                "kernel": jnp.array([[[True, True, False], [False, True, True]], 
                                     [[True, False, False], [False, False, True]]])
            },
            "Dense_1": {
                "bias": jnp.array([[False, True], [False, False]]),
                "kernel": jnp.array([[[True, False], [True, True], [False, True]], 
                                     [[True, False], [False, True], [True, False]]])
            }
        }



        # Combine the masks
        combined_mask = self.packnet.combine_masks(mask_tree, 2)
        expected = {
            "Dense_0": {
                "bias": jnp.array([False, True, True]),
                "kernel": jnp.array([[True, True, False], [False, True, True]])
            },
            "Dense_1": {
                "bias": jnp.array([False, True]),
                "kernel": jnp.array([[True, False], [True, True], [True, True]])
            }
        }
        
        self.assertTrue(jnp.array_equal(
            combined_mask["Dense_0"]["kernel"],
            expected["Dense_0"]["kernel"]
        ), "Dense_0 kernel masks do not match")

        self.assertTrue(jnp.array_equal(
            combined_mask["Dense_1"]["kernel"],
            expected["Dense_1"]["kernel"]
        ), "Dense_1 kernel masks do not match")

        self.seq_length = 3

        mask_tree_more_tasks = {
            "Dense_0": {
                "bias": jnp.array([
                    [False, True, True], 
                    [False, False, False], 
                    [True, True, True]
                ]),
                "kernel": jnp.array([
                    [[True, True, False], [False, True, True]], 
                    [[True, False, False], [False, False, True]],
                    [[True, True, True], [True, True, True]]
                ])
            },
            "Dense_1": {
                "bias": jnp.array([
                    [False, True], 
                    [False, False], 
                    [True, False]
                ]),
                "kernel": jnp.array([
                    [[True, False], [True, True], [False, True]], 
                    [[True, False], [False, True], [False, False]],
                    [[True, True], [True, False], [True, False]]
                    ])
            }
        }
        # Combine the masks
        combined_mask_more_tasks = self.packnet.combine_masks(mask_tree_more_tasks, 2)
        expected_more_tasks = {
            "Dense_0": {
                "bias": jnp.array([False, True, True]),
                "kernel": jnp.array([[True, True, False], [False, True, True]])
            },
            "Dense_1": {
                "bias": jnp.array([False, True]),
                "kernel": jnp.array([[True, False], [True, True], [False, True]])
            }
        }
        self.assertTrue(jnp.array_equal(
            combined_mask_more_tasks["Dense_0"]["kernel"],
            expected_more_tasks["Dense_0"]["kernel"]
        ), "Dense_0 kernel masks do not match")
        self.assertTrue(jnp.array_equal(
            combined_mask_more_tasks["Dense_1"]["kernel"],
            expected_more_tasks["Dense_1"]["kernel"]
        ), "Dense_1 kernel masks do not match")




    # def test_bias_fixing(self):
    #     """Test that bias fixing correctly zeros gradients."""
    #     # Create test gradients
    #     test_grads = {
    #         "Dense_0": {
    #             "kernel": jnp.ones((self.input_size, self.hidden_size)),
    #             "bias": jnp.ones(self.hidden_size)
    #         },
    #         "Dense_1": {
    #             "kernel": jnp.ones((self.hidden_size, self.output_size)),
    #             "bias": jnp.ones(self.output_size)
    #         }
    #     }
        
    #     # Apply fix_biases
    #     fixed_grads = self.packnet.fix_biases(test_grads)
        
    #     # Check that bias gradients are zeroed while kernel gradients are preserved
    #     for layer_name, layer_dict in fixed_grads.items():
    #         if self.packnet.layer_is_prunable(layer_name):
    #             self.assertTrue(jnp.all(layer_dict["bias"] == 0),
    #                           f"Bias gradients for {layer_name} should be zeroed")
    #             self.assertTrue(jnp.all(layer_dict["kernel"] == 1),
    #                           f"Kernel gradients for {layer_name} should be preserved")

    # def test_train_mask(self):
    #     """Test that training mask correctly protects previous task parameters."""
    #     # Set up a scenario with one completed task
    #     self.packnet_state = self.packnet_state.replace(current_task=1)
        
    #     # Create task 0 mask with some parameters assigned to task 0
    #     new_masks = {
    #         0: {
    #             "Dense_0": {
    #                 "kernel": jnp.array([[True, False], [False, True]]),
    #                 "bias": jnp.array([False, False])
    #             }
    #         }
    #     }
    #     self.packnet_state = self.packnet_state.replace(masks=new_masks)
        
    #     # Create test gradients
    #     test_grads = {
    #         "params": {
    #             "Dense_0": {
    #                 "kernel": jnp.ones((2, 2)),
    #                 "bias": jnp.ones(2)
    #             }
    #         }
    #     }
        
    #     # Apply training mask
    #     masked_grads = self.packnet.on_backwards_end(test_grads, self.packnet_state)
        
    #     # Verify gradients for task 0 parameters are zeroed
    #     expected = jnp.array([[0., 1.], [1., 0.]])
    #     self.assertTrue(jnp.array_equal(
    #         masked_grads["params"]["Dense_0"]["kernel"], 
    #         expected
    #     ), "Gradients for task 0 parameters should be zeroed")

    # def test_full_training_cycle(self):
    #     """Test a complete task training and pruning cycle."""
    #     # Create parameters with known values
    #     params = {
    #         "Dense_0": {
    #             "kernel": jnp.array([[0.1, 0.9], [0.8, 0.2]]),
    #             "bias": jnp.array([0.3, 0.4])
    #         }
    #     }
        
    #     # Initial state - task 0, training mode
    #     state = PacknetState(
    #         masks=self.packnet.init_mask_tree(params),
    #         current_task=0,
    #         train_mode=True
    #     )
        
    #     # First, check no masking during initial training
    #     test_grads = {"params": {
    #         "Dense_0": {
    #             "kernel": jnp.ones((2, 2)),
    #             "bias": jnp.ones(2)
    #         }
    #     }}
        
    #     masked_grads = self.packnet.on_backwards_end(test_grads, state)
    #     self.assertTrue(jnp.all(masked_grads["params"]["Dense_0"]["kernel"] == 1),
    #                   "All gradients should be active during initial training")
        
    #     # Now simulate end of training, which should prune
    #     new_params, state = self.packnet.on_train_end(params, state)
        
    #     # State should be in fine-tuning mode
    #     self.assertFalse(state.train_mode, "State should be in fine-tuning mode")
        
    #     # Some parameters should be pruned (exact number depends on implementation)
    #     pruned_count = jnp.sum(jnp.abs(new_params["params"]["Dense_0"]["kernel"]) == 0)
    #     self.assertTrue(pruned_count > 0, "Some parameters should be pruned")
        
    #     # After fine-tuning, simulate moving to next task
    #     _, state = self.packnet.on_finetune_end({}, state)
        
    #     # Should be on task 1 now in training mode
    #     self.assertEqual(state.current_task, 1, "Current task should be 1")
    #     self.assertTrue(state.train_mode, "Should be in training mode")
        
    #     # Task 0 mask should exist and contain True values
    #     self.assertTrue(0 in state.masks, "Task 0 mask should exist")
    #     if 0 in state.masks and "Dense_0" in state.masks[0]:
    #         has_masked = jnp.any(state.masks[0]["Dense_0"]["kernel"])
    #         self.assertTrue(has_masked, "Some parameters should be masked for task 0")

if __name__ == "__main__":
    unittest.main()