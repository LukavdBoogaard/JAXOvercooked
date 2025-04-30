from datetime import datetime

import copy
from datetime import datetime
import pickle
import flax
import jax
import jax.experimental
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.linen.initializers import constant, orthogonal
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from typing import Sequence, NamedTuple, Any, Optional, List
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper

from jax_marl.registration import make
from jax_marl.wrappers.baselines import LogWrapper
from jax_marl.environments.overcooked_environment import overcooked_layouts
from jax_marl.environments.env_selection import generate_sequence
from jax_marl.viz.overcooked_visualizer import OvercookedVisualizer
from dotenv import load_dotenv
import os
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import wandb
from functools import partial
from dataclasses import dataclass, field
import tyro
from tensorboardX import SummaryWriter
from pathlib import Path

from collections import defaultdict


import pickle

class ContinualLearningMetrics:
    def __init__(self):
        # Store performance by task index and global step
        self.performance_history = defaultdict(dict)  # {task_idx: {step: [perf_task1, perf_task2, ...]}}
        self.random_performance = {}
        self.optimal_performance = {}
        
    def record_evaluation(self, current_task_idx, global_step, evaluations):
        """Record evaluation results for all tasks at current step"""
        self.performance_history[current_task_idx][global_step] = evaluations
        
    def set_baselines(self, random_perf, optimal_perf):
        """Set baseline performances for normalization"""
        self.random_performance = random_perf
        self.optimal_performance = optimal_perf
    
    def normalize_score(self, task_idx, score):
        """Normalize score to [0,1] range as per paper"""
        if task_idx not in self.random_performance or task_idx not in self.optimal_performance:
            return score  # Return raw score if baselines not available
        
        random = self.random_performance[task_idx]
        optimal = self.optimal_performance[task_idx]
        denominator = optimal - random
        
        if abs(denominator) < 1e-8:
            return 0.0  # Avoid division by zero
            
        return (score - random) / denominator
    
    def calculate_average_performance(self, num_tasks, task_iterations):
        """Calculate average performance (P) across all tasks over time (Eq 1)"""
        total_steps = num_tasks * task_iterations
        total_success = 0.0
        count = 0
        
        # For each task's training period
        for task_idx in range(num_tasks):
            # Get all evaluation points for this task
            steps = sorted(self.performance_history[task_idx].keys())
            
            for step in steps:
                all_task_perf = self.performance_history[task_idx][step]
                normalized_perf = [self.normalize_score(i, perf) for i, perf in enumerate(all_task_perf)]
                avg_success = sum(normalized_perf) / len(normalized_perf)
                total_success += avg_success
                count += 1
        
        if count == 0:
            return 0.0
        return total_success / count
    
    def calculate_forgetting(self, num_tasks, k=5):
        """Calculate forgetting (F) for all tasks except last (Eq 2)"""
        if num_tasks <= 1:
            return 0.0  # No forgetting possible with single task
            
        forgetting_values = []
        
        # For each task except the last one
        for task_idx in range(num_tasks - 1):
            # Find end of training for this task
            this_task_steps = sorted(self.performance_history[task_idx].keys())
            if not this_task_steps:
                continue
                
            # Get k last evaluations of this task during its training
            end_of_training_steps = this_task_steps[-k:]
            end_of_training_perf = []
            for step in end_of_training_steps:
                if task_idx < len(self.performance_history[task_idx][step]):
                    end_of_training_perf.append(self.normalize_score(
                        task_idx, self.performance_history[task_idx][step][task_idx]))
            
            if not end_of_training_perf:
                continue
            
            # Get k last evaluations at end of sequence
            last_task_steps = sorted(self.performance_history[num_tasks-1].keys())
            if not last_task_steps:
                continue
                
            end_of_sequence_steps = last_task_steps[-k:]
            end_of_sequence_perf = []
            for step in end_of_sequence_steps:
                if task_idx < len(self.performance_history[num_tasks-1][step]):
                    end_of_sequence_perf.append(self.normalize_score(
                        task_idx, self.performance_history[num_tasks-1][step][task_idx]))
            
            if not end_of_sequence_perf:
                continue
                
            # Calculate forgetting
            avg_end_training = sum(end_of_training_perf) / len(end_of_training_perf)
            avg_end_sequence = sum(end_of_sequence_perf) / len(end_of_sequence_perf)
            task_forgetting = avg_end_training - avg_end_sequence
            forgetting_values.append(task_forgetting)
            
        # Average forgetting across tasks
        if not forgetting_values:
            return 0.0
        return sum(forgetting_values) / len(forgetting_values)
        
    def calculate_forward_transfer(self, num_tasks):
        """Calculate forward transfer (FT) for all tasks except first (Eq 3)"""
        if num_tasks <= 1:
            return 0.0  # No forward transfer with single task
        
        forward_transfer_values = []
        
        # Skip first task, no previous knowledge to transfer
        for task_idx in range(1, num_tasks):
            # Get AUC for task in continual learning setting
            task_steps = sorted(self.performance_history[task_idx].keys())
            if not task_steps:
                continue
                
            task_perf = []
            for step in task_steps:
                if task_idx < len(self.performance_history[task_idx][step]):
                    task_perf.append(self.normalize_score(
                        task_idx, self.performance_history[task_idx][step][task_idx]))
            
            if not task_perf:
                continue
                
            # Calculate AUC (approximate as average performance)
            cl_auc = sum(task_perf) / len(task_perf)
            
            # For simplicity, assume baseline AUC is 0.5 (or use your own baseline)
            # In a real implementation, you would have separate baseline runs
            baseline_auc = 0.5
            
            # Normalized difference
            ft = (cl_auc - baseline_auc) / (1.0 - baseline_auc + 1e-8)
            forward_transfer_values.append(ft)
            
        if not forward_transfer_values:
            return 0.0
        return sum(forward_transfer_values) / len(forward_transfer_values)
