from scipy.optimize import minimize

result = minimize(lambda x: (x[0] - 2) ** 2 + (x[1] - 3) ** 2, x0=[0, 0])
print('最优解：', result.x)
print('目标函数值：', result.fun)
