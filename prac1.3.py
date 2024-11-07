from scipy.optimize import minimize
from scipy import stats, interpolate
import numpy as np

print("IU2141230160")

def rosen(x):
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

res = minimize(rosen, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
print("Optimized parameters:", res.x)

data = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

mean = np.mean(data)
std_dev = np.std(data)
print("Mean:", mean)
print("Standard Deviation:", std_dev)

t_stat, p_value = stats.ttest_1samp(data, 5)
print("T-statistic:", t_stat)
print("P-value:", p_value)

x = np.linspace(0, 10, 10)
y = np.sin(x)
f_interp = interpolate.interp1d(x, y, kind='linear')
x_new = 2.5
y_new = f_interp(x_new)
print(f"Interpolated value at x={x_new}: {y_new}")
