# src/core/orchestrator.py
"""
Orchestrator module for coordinating geometry, physics, and optimizer.
"""

import json
import time
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Standard optimization result container."""
    best_solution: List[float]
    best_fitness: float
    history: List[dict]
    elapsed_time: float
    status: str = "Success"
    optimizer_state: Optional[dict] = None


class OptimizationSystem:
    """
    Orchestrator for blade optimization.
    Coordinates: Geometry -> Physics -> Optimizer
    """
    
    def __init__(self,
                 geometry: Any = None,
                 physics: Any = None,
                 optimizer: Any = None,
                 save_dir: str = "./results",
                 fitness_extractor: Optional[Callable[[Any], float]] = None):
        """
        Parameters
        ----------
        geometry : object
            Geometry module with build() method
        physics : object
            Physics module with solve() method
        optimizer : object
            Optimizer with set_objective() and optimize() methods
        save_dir : str
            Directory to save results
        fitness_extractor : callable, optional
            Function to extract fitness from physics result
        """
        self.geo = geometry
        self.phys = physics
        self.opt = optimizer
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.fitness_extractor = fitness_extractor
        self._history: List[dict] = []
        self._start_time = 0.0
    
    def _extract_fitness(self, physics_result: Any) -> float:
        """Extract fitness value from physics result."""
        if self.fitness_extractor is not None:
            return float(self.fitness_extractor(physics_result))
        if hasattr(physics_result, 'fitness'):
            return float(physics_result.fitness)
        try:
            return float(physics_result)
        except TypeError:
            raise TypeError(
                "Physics result cannot be converted to float. "
                "Please provide fitness_extractor."
            )
    
    def _objective_wrapper(self, params: List[float]) -> float:
        """Wrapper: parameters -> geometry -> physics -> fitness."""
        try:
            geometry = self.geo.build_from_vector(params)
            physics_result = self.phys.solve(geometry)
            fitness = self._extract_fitness(physics_result)
            self._history.append({
                "params": params,
                "fitness": fitness,
                "timestamp": time.time()
            })
            return fitness
        except Exception as e:
            logger.error(f"Error evaluating parameters {params}: {e}")
            return float('inf')
    
    def run(self, n_pop: int, n_iters: int, bounds: List[List[float]],
            seed: int = 42, reset_history: bool = True) -> OptimizationResult:
        """Run complete optimization."""
        logger.info("🚀 Starting integrated optimization...")
        if reset_history:
            self._history = []
        
        self._start_time = time.time()
        status = "Success"
        best_sol = []
        best_fit = float('inf')
        optimizer_state = None
        
        try:
            self.opt.set_objective(self._objective_wrapper)
            raw_output = self.opt.optimize(
                n_pop=n_pop,
                n_iters=n_iters,
                bounds=bounds,
                seed=seed
            )
            
            if isinstance(raw_output, dict):
                best_sol = raw_output.get('solution', [])
                best_fit = raw_output.get('fitness', float('inf'))
            else:
                best_sol = getattr(raw_output, 'solution', [])
                best_fit = getattr(raw_output, 'fitness', float('inf'))
            
            if hasattr(self.opt, 'get_state'):
                try:
                    optimizer_state = self.opt.get_state()
                except Exception as e:
                    logger.warning(f"Could not get optimizer state: {e}")
        
        except Exception as e:
            logger.error(f"❌ Critical error in optimization: {e}")
            status = "Failed"
        
        elapsed = time.time() - self._start_time
        
        result = OptimizationResult(
            best_solution=best_sol,
            best_fitness=best_fit,
            history=self._history,
            elapsed_time=elapsed,
            status=status,
            optimizer_state=optimizer_state,
        )
        
        self._save_result(result)
        logger.info(f"✅ Optimization complete. Status: {status} | Best fitness: {best_fit}")
        return result
    
    def _save_result(self, result: OptimizationResult):
        """Save results to JSON."""
        file_path = self.save_dir / f"result_{int(time.time())}.json"
        data = asdict(result)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4, default=str)
        logger.info(f"💾 Results saved: {file_path}")
    
    def resume(self, state_file: str):
        """Resume optimization from checkpoint."""
        state_path = Path(state_file)
        if not state_path.exists():
            logger.error(f"State file not found: {state_file}")
            return
        
        with open(state_path, 'r') as f:
            data = json.load(f)
        
        optimizer_state = data.get('optimizer_state')
        if optimizer_state is None:
            logger.warning("No optimizer_state found in file.")
        else:
            if hasattr(self.opt, 'set_state'):
                self.opt.set_state(optimizer_state)
                logger.info("♻️ Optimizer state restored.")
            else:
                logger.warning("Optimizer does not have set_state() method.")
        
        history = data.get('history', [])
        if history:
            self._history = history
            logger.info(f"📜 Restored {len(history)} historical evaluations.")