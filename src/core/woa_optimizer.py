# src/core/woa_optimizer.py
"""
Whale Optimization Algorithm (WOA) for blade optimization.
Compatible with Orchestrator protocol.
"""

import numpy as np
import json
import os
import logging
from typing import Callable, List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhaleOptimizationOptimizer:
    """
    WOA optimizer with checkpoint and early stopping.
    
    Usage:
        optimizer = WhaleOptimizationOptimizer()
        optimizer.set_objective(func)
        result = optimizer.optimize(n_pop=30, n_iters=100, bounds=bounds, seed=42)
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints",
                 early_stop_patience: int = 10):
        self.checkpoint_dir = checkpoint_dir
        self.early_stop_patience = early_stop_patience
        self.objective_func = None
        self.population = None
        self.best_pos = None
        self.best_score = float('inf')
        self.convergence_history = []
        self.iteration = 0
        self._no_improvement_count = 0
        self._bounds = None
        self._pop_size = None
        self._max_iters = None
        self._seed = None
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
    def set_objective(self, func: Callable[[np.ndarray], float]):
        """Set objective function for minimization."""
        self.objective_func = func
    
    def _initialize_population(self):
        if self._bounds is None or self._pop_size is None:
            raise ValueError("Bounds and population size must be set before initialization.")
        self.population = np.zeros((self._pop_size, len(self._bounds)))
        for i, (low, high) in enumerate(self._bounds):
            self.population[:, i] = np.random.uniform(low, high, self._pop_size)
    
    def _evaluate(self, population: np.ndarray) -> np.ndarray:
        """Evaluate objective function for population."""
        return np.array([self.objective_func(ind) for ind in population])
    
    def _early_stop(self) -> bool:
        if len(self.convergence_history) < self.early_stop_patience:
            return False
        recent = self.convergence_history[-self.early_stop_patience:]
        return np.all(np.diff(recent) >= 0)
    
    def optimize(self, n_pop: int, n_iters: int,
                 bounds: List[List[float]], seed: int) -> Dict[str, Any]:
        """
        Run WOA optimization.
        
        Returns
        -------
        dict with keys: solution, fitness, history
        """
        if self.objective_func is None:
            raise ValueError("Objective function not set. Call set_objective() first.")
        
        self._pop_size = n_pop
        self._max_iters = n_iters
        self._bounds = np.array(bounds)
        self._seed = seed
        
        np.random.seed(seed)
        self._initialize_population()
        
        # Evaluate initial population
        scores = self._evaluate(self.population)
        min_idx = np.argmin(scores)
        self.best_score = scores[min_idx]
        self.best_pos = np.copy(self.population[min_idx])
        self.convergence_history = [self.best_score]
        self.iteration = 0
        self._no_improvement_count = 0
        
        for it in range(1, n_iters + 1):
            self.iteration = it
            a = 2.0 * (1.0 - it / n_iters)
            
            # Update population
            for i in range(n_pop):
                r1, r2 = np.random.rand(), np.random.rand()
                A = 2 * a * r1 - a
                C = 2 * r2
                p = np.random.rand()
                
                if p < 0.5:
                    if abs(A) < 1:
                        # Encircling prey
                        D = np.abs(C * self.best_pos - self.population[i])
                        new_pos = self.best_pos - A * D
                    else:
                        # Random search
                        rand_idx = np.random.randint(n_pop)
                        D = np.abs(C * self.population[rand_idx] - self.population[i])
                        new_pos = self.population[rand_idx] - A * D
                else:
                    # Bubble-net attack
                    dist = np.abs(self.best_pos - self.population[i])
                    l = np.random.uniform(-1, 1)
                    new_pos = dist * np.exp(l) * np.cos(2 * np.pi * l) + self.best_pos
                
                # Apply bounds
                new_pos = np.clip(new_pos, self._bounds[:, 0], self._bounds[:, 1])
                self.population[i] = new_pos
            
            # Evaluate new population
            scores = self._evaluate(self.population)
            min_idx = np.argmin(scores)
            if scores[min_idx] < self.best_score:
                self.best_score = scores[min_idx]
                self.best_pos = np.copy(self.population[min_idx])
                self._no_improvement_count = 0
            else:
                self._no_improvement_count += 1
            
            self.convergence_history.append(self.best_score)
            
            # Save checkpoint
            if it % 10 == 0:
                self._save_checkpoint()
            
            # Early stopping
            if self._early_stop():
                logger.info(f"Early stopping at iteration {it}")
                break
        
        return {
            'solution': self.best_pos.tolist(),
            'fitness': self.best_score,
            'history': self.convergence_history
        }
    
    def _save_checkpoint(self):
        """Save optimizer state."""
        state = self.get_state()
        filename = os.path.join(self.checkpoint_dir, "checkpoint_latest.json")
        with open(filename, 'w') as f:
            json.dump(state, f, indent=4)
        logger.info(f"Checkpoint saved to {filename}")
    
    def get_state(self) -> Dict[str, Any]:
        """Return optimizer state for resuming."""
        return {
            'iteration': self.iteration,
            'best_pos': self.best_pos.tolist() if self.best_pos is not None else None,
            'best_score': self.best_score,
            'population': self.population.tolist() if self.population is not None else None,
            'convergence_history': self.convergence_history,
            'bounds': self._bounds.tolist() if self._bounds is not None else None,
            'pop_size': self._pop_size,
            'max_iters': self._max_iters,
            'seed': self._seed,
            'no_improvement_count': self._no_improvement_count
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Restore optimizer state."""
        self.iteration = state.get('iteration', 0)
        self.best_score = state.get('best_score', float('inf'))
        self.best_pos = np.array(state['best_pos']) if state.get('best_pos') else None
        self.population = np.array(state['population']) if state.get('population') else None
        self.convergence_history = state.get('convergence_history', [])
        self._bounds = np.array(state['bounds']) if state.get('bounds') else None
        self._pop_size = state.get('pop_size')
        self._max_iters = state.get('max_iters')
        self._seed = state.get('seed')
        self._no_improvement_count = state.get('no_improvement_count', 0)