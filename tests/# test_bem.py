# test_bem.py
"""
Simple test script for BEM solver with blade geometry.
Run from project root: python3 test_bem.py
"""

import sys
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

from geometry.blade_geometry import BladeGeometry
from aero.bem_solver import BEMSolver


def main():
    print("🔧 Building blade geometry...")
    blade = BladeGeometry(R=5.0, hub_radius=0.8, num_stations=30)
    
    # Design vector: [chord1..4, twist1..4]
    design = [0.35, 0.30, 0.25, 0.15, 25.0, 20.0, 12.0, 5.0]
    data = blade.build_from_vector(design)
    print(f"✅ Geometry valid: {data['valid']}")
    
    print("\n🔧 Running BEM solver...")
    solver = BEMSolver(R=5.0, B=3)
    results = solver.solve(data, wind_speed=8.0, rpm=110)
    
    # Calculate tip speed ratio
    omega = 110 * 2 * np.pi / 60
    TSR = omega * 5.0 / 8.0
    
    print("\n📊 Results:")
    print(f"   Power:  {results['power']/1000:.2f} kW")
    print(f"   Thrust: {results['thrust']:.1f} N")
    print(f"   Torque: {results['torque']:.1f} N·m")
    print(f"   Max axial induction: {results['axial_induction'].max():.3f}")
    print(f"   Tip speed ratio: {TSR:.2f}")
    
    # Check if power is reasonable
    if results['power'] > 8000:
        print("   ✅ Power is above 8 kW (good for this geometry)")
    else:
        print("   ⚠️ Power is low; geometry needs optimization")
    
    return results


if __name__ == "__main__":
    main()