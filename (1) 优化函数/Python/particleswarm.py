from scipy.optimize import differential_evolution

def objective(x):
    return (x[0] - 2) ** 2 + (x[1] - 3) ** 2

bounds = [(0, 5), (0, 5)]

result = differential_evolution(objective, bounds)
print('最优解：', result.x)
print('目标函数值：', result.fun)
