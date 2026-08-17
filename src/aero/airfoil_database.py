import numpy as np
import pandas as pd

class AirfoilDatabase:
    def __init__(self, csv_path=None):
        if csv_path is None:
            # داده‌های Mock برای تست (NACA 4412)
            alphas = np.linspace(-5, 20, 10)
            res = [300000, 500000, 1000000]
            self.data = {}
            for Re in res:
                cl = 0.1 * alphas + 0.3  # رابطه خطی ساده
                cd = 0.008 + 0.0001 * (alphas + 2)**2
                self.data[Re] = {'alpha': alphas, 'cl': cl, 'cd': cd}
        else:
            self.data = pd.read_csv(csv_path)
    
    def get_coeffs(self, alpha, Re):
        """درون‌یابی دوبعدی برای یافتن cl و cd"""
        # پیدا کردن نزدیک‌ترین Re
        Re_list = sorted(self.data.keys())
        Re_low = max([r for r in Re_list if r <= Re], default=Re_list[0])
        Re_high = min([r for r in Re_list if r >= Re], default=Re_list[-1])
        
        if Re_low == Re_high:
            Re_data = self.data[Re_low]
            cl = np.interp(alpha, Re_data['alpha'], Re_data['cl'])
            cd = np.interp(alpha, Re_data['alpha'], Re_data['cd'])
        else:
            # درون‌یابی خطی بین دو Re
            low_data = self.data[Re_low]
            high_data = self.data[Re_high]
            cl_low = np.interp(alpha, low_data['alpha'], low_data['cl'])
            cd_low = np.interp(alpha, low_data['alpha'], low_data['cd'])
            cl_high = np.interp(alpha, high_data['alpha'], high_data['cl'])
            cd_high = np.interp(alpha, high_data['alpha'], high_data['cd'])
            weight = (Re - Re_low) / (Re_high - Re_low)
            cl = cl_low + weight * (cl_high - cl_low)
            cd = cd_low + weight * (cd_high - cd_low)
        
        return cl, cd
