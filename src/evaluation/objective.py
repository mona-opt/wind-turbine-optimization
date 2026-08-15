import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.geometry.blade_geometry import BladeGeometry
from src.evaluation.performance_analyzer import PerformanceAnalyzer

class ObjectiveFunction:
    def __init__(self, R=5.0, hub_radius=0.8, num_stations=30, num_blades=3,
                 air_density=1.225, weibull_k=1.3, weibull_c=8.0,
                 cut_in=3.0, cut_out=25.0, penalty_weight=1e6):
        self.R = R
        self.hub_radius = hub_radius
        self.num_stations = num_stations
        self.rho = air_density
        self.weibull_k = weibull_k
        self.weibull_c = weibull_c
        self.cut_in = cut_in
        self.cut_out = cut_out
        self.penalty_weight = penalty_weight
        self.performance = PerformanceAnalyzer(R=R, B=num_blades, rho=air_density)
        self.wind_speeds = np.linspace(cut_in, cut_out, 20)
        k, c = weibull_k, weibull_c
        pdf = (k/c)*(self.wind_speeds/c)**(k-1)*np.exp(-(self.wind_speeds/c)**k)
        self.weights = pdf * (self.wind_speeds[1] - self.wind_speeds[0])
    
    def evaluate(self, x):
        blade = BladeGeometry(R=self.R, hub_radius=self.hub_radius, num_stations=self.num_stations)
        try:
            geo = blade.build_from_vector(x)
        except:
            return 1e9
        if not geo['valid']:
            return 1e9
        try:
            power = self.performance.get_power_curve(geo, self.wind_speeds)
        except:
            return 1e9
        if np.max(power) < 100:
            return 1e9
        power = np.maximum(power, 0)
        avg_power = np.trapezoid(power * self.weights, self.wind_speeds)
        aep = avg_power * 8760 / 1000
        aep = np.clip(aep, 0, 100000)
        return -aep
    
    def evaluate_with_details(self, x):
        blade = BladeGeometry(R=self.R, hub_radius=self.hub_radius, num_stations=self.num_stations)
        geo = blade.build_from_vector(x)
        power = self.performance.get_power_curve(geo, self.wind_speeds)
        power = np.maximum(power, 0)
        avg_power = np.trapezoid(power * self.weights, self.wind_speeds)
        aep = avg_power * 8760 / 1000
        aep = np.clip(aep, 0, 100000)
        return {'geometry': geo, 'power_curve': power, 'aep_kwh': aep, 'objective': -aep}
