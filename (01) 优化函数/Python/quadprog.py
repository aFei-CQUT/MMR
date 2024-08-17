import numpy as np
from scipy.optimize import minimize

H = np.array([[1, 0], [0, 1]])  # 二次项系数矩阵
C = np.array([-1, -2])  # 线性项系数向量
A = np.array([[1, 1], [-1, 2], [2, 1]])  # 不等式约束矩阵
b = np.array([2, 2, 3])  # 不等式约束向量
bounds = [(0, None), (0, None)]  # 变量界限

def objective(x):
    return 0.5 * x.T @ H @ x + C.T @ x

result = minimize(objective, x0=np.zeros(2), bounds=bounds, constraints={'type': 'ineq', 'fun': lambda x: b - A @ x})
print('最优解：', result.x)
print('目标函数值：', result.fun)
