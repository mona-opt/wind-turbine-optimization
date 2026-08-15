# test_bem.py
"""
Simple test script for BEM solver with blade geometry.
Run from project root: python3 test_bem.py
"""

import sys
import numpy as np

# Add src to path so we can import modules
sys.path.insert(0, 'src')

from geometry.blade_geometry import BladeGeometry
from aero.bem_solver import BEMSolver


def main():
    print("🔧 Building blade geometry...")
    
    # Create blade geometry with 30 radial stations
    blade = BladeGeometry(R=5.0, hub_radius=0.8, num_stations=30)
    
    # Design vector: [chord1, chord2, chord3, chord4, twist1, twist2, twist3, twist4]
    # Control points at r/R = [0.20, 0.45, 0.70, 0.95]
    design = [0.35, 0.30, 0.25, 0.15, 25.0, 20.0, 12.0, 5.0]
    
    # Build geometry from design vector
    data = blade.build_from_vector(design)
    print(f"✅ Geometry valid: {data['valid']}")
    
    if not data['valid']:
        print("⚠️ Geometry validation failed:")
        for msg in data['messages']:
            print(f"   - {msg}")
        return
    
    print("\n🔧 Running BEM solver...")
    
    # Initialize BEM solver
    solver = BEMSolver(R=5.0, B=3, air_density=1.225)
    
    # Solve at design point: V=8 m/s, RPM=110
    results = solver.solve(data, wind_speed=8.0, rpm=110)
    
    # Calculate tip speed ratio
    omega = 110 * 2 * np.pi / 60.0   # rad/s
    TSR = omega * 5.0 / 8.0           # lambda = omega*R / V
    
    # Print results
    print("\n📊 Results:")
    print(f"   Power:              {results['power']/1000:.2f} kW")
    print(f"   Thrust:             {results['thrust']:.1f} N")
    print(f"   Torque:             {results['torque']:.1f} N·m")
    print(f"   Max axial induction: {results['axial_induction'].max():.3f}")
    print(f"   Tip speed ratio:    {TSR:.2f}")
    
    # Check if power is reasonable for 10 kW turbine
    if results['power'] >= 8000:
        print("   ✅ Power is above 8 kW (good for this geometry)")
    elif results['power'] >= 5000:
        print("   ⚠️ Power is moderate (5-8 kW); geometry needs improvement")
    else:
        print("   ⚠️ Power is low (< 5 kW); geometry needs significant optimization")
    
    # Check axial induction
    if results['axial_induction'].max() > 0.5:
        print("   ⚠️ High axial induction (> 0.5); check Glauert correction")
    
    return results


if __name__ == "__main__":
    main()
