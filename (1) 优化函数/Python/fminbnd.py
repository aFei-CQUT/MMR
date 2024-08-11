from scipy.optimize import minimize_scalar

result = minimize_scalar(lambda x: (x - 2) ** 2 + 1, bounds=(0, 4), method='bounded')
print('最优解：', result.x)
print('目标函数值：', result.fun)
