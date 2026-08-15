import sys
import os
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.objective import ObjectiveFunction
from src.core.woa_optimizer import WhaleOptimizationOptimizer


def main():
    print("=" * 50)
    print("⚡ LIGHT OPTIMIZATION TEST (5 pop, 5 iters)")
    print("=" * 50)
    
    R = 5.0
    hub_radius = 0.8
    num_stations = 20
    num_blades = 3
    
    bounds = [
        [0.05, 0.60], [0.05, 0.60], [0.05, 0.60], [0.05, 0.60],
        [0.0, 35.0], [0.0, 35.0], [0.0, 35.0], [0.0, 35.0]
    ]
    
    objective = ObjectiveFunction(
        R=R,
        hub_radius=hub_radius,
        num_stations=num_stations,
        num_blades=num_blades,
        weibull_k=1.3,
        weibull_c=8.0,
        cut_in=3.0,
        cut_out=25.0
    )
    
    # کاهش سرعت‌های باد برای تست سبک
    objective.wind_speeds = np.linspace(3, 25, 10)
    # محاسبه وزن‌ها با فرمول مستقیم (بدون متد جداگانه)
    k, c = objective.weibull_k, objective.weibull_c
    pdf = (k/c) * (objective.wind_speeds/c)**(k-1) * np.exp(-(objective.wind_speeds/c)**k)
    objective.weights = pdf * (objective.wind_speeds[1] - objective.wind_speeds[0])
    
    initial_design = [0.35, 0.30, 0.25, 0.15, 25.0, 20.0, 12.0, 5.0]
    init_result = objective.evaluate_with_details(initial_design)
    print(f"\n📊 Initial AEP: {init_result['aep_kwh']:.0f} kWh/year")
    
    optimizer = WhaleOptimizationOptimizer(
        checkpoint_dir="checkpoints_light",
        early_stop_patience=3
    )
    
    def obj_wrapper(x):
        return objective.evaluate(np.array(x))
    
    optimizer.set_objective(obj_wrapper)
    
    print(f"\n🎯 Running WOA with:")
    print(f"   Population: 5")
    print(f"   Iterations: 5")
    print(f"   Seed: 42")
    
    start = time.time()
    result = optimizer.optimize(
        n_pop=5,
        n_iters=5,
        bounds=bounds,
        seed=42
    )
    elapsed = time.time() - start
    
    best_sol = result['solution']
    best_fit = result['fitness']
    
    print("\n" + "=" * 50)
    print("📊 RESULTS")
    print("=" * 50)
    print(f"✅ Best AEP: {-best_fit:.0f} kWh/year")
    print(f"   Time: {elapsed:.1f} seconds")
    print(f"\n📋 Best design:")
    print(f"   Chord: [{best_sol[0]:.3f}, {best_sol[1]:.3f}, {best_sol[2]:.3f}, {best_sol[3]:.3f}] m")
    print(f"   Twist: [{best_sol[4]:.1f}, {best_sol[5]:.1f}, {best_sol[6]:.1f}, {best_sol[7]:.1f}] deg")
    
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/light_test_result.json", "w") as f:
        json.dump({
            'best_solution': best_sol,
            'best_aep': -best_fit,
            'time': elapsed
        }, f, indent=2)
    print("\n💾 Saved to outputs/light_test_result.json")


if __name__ == "__main__":
    main()
