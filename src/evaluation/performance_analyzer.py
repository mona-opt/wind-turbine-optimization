import numpy as np

class PerformanceAnalyzer:
    def __init__(self, R: float, B: int = 3, rho: float = 1.225):
        self.R = R
        self.B = B
        self.rho = rho
        self.rated_power = 6000  # کاهش به ۶ کیلووات برای AEP واقعی‌تر
        self.rated_wind = 8.0
    
    def get_power_curve(self, geometry_data, wind_speeds, rpm_strategy='variable'):
        power_curve = []
        avg_chord = np.mean(geometry_data['chord'])
        avg_twist = np.mean(geometry_data['twist'])
        chord_factor = min(1.0, avg_chord / 0.3)
        twist_factor = 1.0 - 0.01 * (avg_twist - 15)**2 / 100
        twist_factor = max(0.5, min(1.0, twist_factor))
        
        for v in wind_speeds:
            if v < 3.0:
                power = 0.0
            elif v < self.rated_wind:
                power = self.rated_power * (v / self.rated_wind)**3
            elif v < 25.0:
                power = self.rated_power
            else:
                power = 0.0
            power *= chord_factor * twist_factor
            power_curve.append(max(0, power))
        
        return np.array(power_curve)
