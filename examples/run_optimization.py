# examples/run_optimization.py
"""
Run complete blade optimization using WOA.
Project: 10 kW HAWT for Manjil site.
Optimizes 8 design variables: [chord1..4, twist1..4]
Maximizes Annual Energy Production (AEP) with Weibull distribution.
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.objective import ObjectiveFunction
from src.core.woa_optimizer import WhaleOptimizationOptimizer
from src.core.orchestrator import OptimizationSystem


def main():
    print("=" * 60)
    print("🚀 10 kW Wind Turbine Blade Optimization - Manjil Site")
    print("=" * 60)
    
    # ===================================================================
    # 1. Define problem parameters
    # ===================================================================
    R = 5.0
    hub_radius = 0.8
    num_stations = 30
    num_blades = 3
    
    # Design variable bounds: [chord1..4, twist1..4]
    # Control points at r/R = [0.20, 0.45, 0.70, 0.95]
    bounds = [
        [0.05, 0.60],   # chord 1 (m)
        [0.05, 0.60],   # chord 2 (m)
        [0.05, 0.60],   # chord 3 (m)
        [0.05, 0.60],   # chord 4 (m)
        [0.0, 35.0],    # twist 1 (deg)
        [0.0, 35.0],    # twist 2 (deg)
        [0.0, 35.0],    # twist 3 (deg)
        [0.0, 35.0]     # twist 4 (deg)
    ]
    
    # WOA parameters
    n_pop = 20
    n_iters = 30
    seed = 42
    
    # ===================================================================
    # 2. Create objective function
    # ===================================================================
    print("\n📋 Initializing objective function...")
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
    
    # ===================================================================
    # 3. Test initial design
    # ===================================================================
    print("\n🔍 Testing initial design...")
    initial_design = [0.35, 0.30, 0.25, 0.15, 25.0, 20.0, 12.0, 5.0]
    init_result = objective.evaluate_with_details(initial_design)
    print(f"   Initial AEP: {init_result['aep_kwh']:.0f} kWh/year")
    print(f"   Violations: {len(init_result['violations'])}")
    
    # ===================================================================
    # 4. Create optimizer
    # ===================================================================
    print("\n🔧 Initializing WOA optimizer...")
    optimizer = WhaleOptimizationOptimizer(
        checkpoint_dir="checkpoints",
        early_stop_patience=10
    )
    
    # ===================================================================
    # 5. Create orchestrator and run optimization
    # ===================================================================
    print("\n🎯 Starting optimization...")
    print(f"   Population: {n_pop}")
    print(f"   Iterations: {n_iters}")
    print(f"   Seed: {seed}")
    
    system = OptimizationSystem(
        geometry=None,  # Not used directly
        physics=None,   # Not used directly
        optimizer=optimizer,
        save_dir="outputs",
        fitness_extractor=None
    )
    
    # Override the objective wrapper
    def objective_wrapper(x):
        return objective.evaluate(np.array(x))
    
    optimizer.set_objective(objective_wrapper)
    
    # Run optimization
    start_time = time.time()
    result = optimizer.optimize(
        n_pop=n_pop,
        n_iters=n_iters,
        bounds=bounds,
        seed=seed
    )
    elapsed = time.time() - start_time
    
    # ===================================================================
    # 6. Display results
    # ===================================================================
    print("\n" + "=" * 60)
    print("📊 OPTIMIZATION RESULTS")
    print("=" * 60)
    
    best_solution = result['solution']
    best_fitness = result['fitness']
    
    print(f"\n✅ Best fitness (negative AEP): {best_fitness:.0f}")
    print(f"   Best AEP: {-best_fitness:.0f} kWh/year")
    print(f"   Time: {elapsed:.1f} seconds")
    
    # Evaluate best design in detail
    print("\n📋 Best design vector:")
    print(f"   Chord: [{best_solution[0]:.3f}, {best_solution[1]:.3f}, "
          f"{best_solution[2]:.3f}, {best_solution[3]:.3f}] m")
    print(f"   Twist: [{best_solution[4]:.1f}, {best_solution[5]:.1f}, "
          f"{best_solution[6]:.1f}, {best_solution[7]:.1f}] deg")
    
    # Detailed evaluation
    best_details = objective.evaluate_with_details(best_solution)
    print(f"\n📈 Detailed performance:")
    print(f"   AEP: {best_details['aep_kwh']:.0f} kWh/year")
    print(f"   Penalty: {best_details['penalty']:.0f}")
    print(f"   Violations: {len(best_details['violations'])}")
    
    if best_details['violations']:
        print("   ⚠️ Violations:")
        for v in best_details['violations']:
            print(f"      - {v}")
    
    # ===================================================================
    # 7. Save results
    # ===================================================================
    print("\n💾 Saving results...")
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    
    # Save JSON
    output_data = {
        'best_solution': best_solution,
        'best_fitness': best_fitness,
        'best_aep_kwh': -best_fitness,
        'n_pop': n_pop,
        'n_iters': n_iters,
        'seed': seed,
        'elapsed_time': elapsed,
        'bounds': bounds,
        'history': result.get('history', []),
        'geometry': {
            'r': best_details['geometry']['r'].tolist(),
            'chord': best_details['geometry']['chord'].tolist(),
            'twist': best_details['geometry']['twist'].tolist()
        },
        'performance': {
            'power_curve': best_details['power_curve'].tolist(),
            'aep_kwh': best_details['aep_kwh'],
            'violations': best_details['violations']
        }
    }
    
    with open("outputs/optimization_results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    print("   ✅ Results saved to outputs/optimization_results.json")
    
    # ===================================================================
    # 8. Generate blade geometry CSV
    # ===================================================================
    print("\n📁 Generating blade geometry CSV...")
    from src.geometry.blade_geometry import BladeGeometry
    
    blade = BladeGeometry(R=R, hub_radius=hub_radius, num_stations=num_stations)
    geo_data = blade.build_from_vector(best_solution)
    
    import csv
    with open("outputs/best_blade_geometry.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["r (m)", "r/R", "chord (m)", "twist (deg)"])
        for i in range(len(geo_data['r'])):
            writer.writerow([
                geo_data['r'][i],
                geo_data['r'][i] / R,
                geo_data['chord'][i],
                geo_data['twist'][i]
            ])
    print("   ✅ Geometry saved to outputs/best_blade_geometry.csv")
    
    # ===================================================================
    # 9. Summary
    # ===================================================================
    print("\n" + "=" * 60)
    print("✅ OPTIMIZATION COMPLETE!")
    print("=" * 60)
    print(f"\n   Best AEP: {-best_fitness:.0f} kWh/year")
    print(f"   Improvement: {-best_fitness - init_result['aep_kwh']:.0f} kWh/year")
    print(f"   Improvement (%): {(-best_fitness / init_result['aep_kwh'] - 1) * 100:.1f}%")
    print("\n📁 Output files:")
    print("   - outputs/optimization_results.json")
    print("   - outputs/best_blade_geometry.csv")
    print("   - outputs/figures/ (optional, for plots)")


if __name__ == "__main__":
    main()