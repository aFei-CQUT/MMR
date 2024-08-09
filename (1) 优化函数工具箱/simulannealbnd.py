from scipy.optimize import dual_annealing

def objective(x):
    return (x - 2) ** 2 + 1

bounds = [(-5, 5)]

result = dual_annealing(objective, bounds)
print('最优解：', result.x)
print('目标函数值：', result.fun)
