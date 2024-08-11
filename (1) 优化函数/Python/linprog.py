from scipy.optimize import linprog

c = [-1, -2]  # 目标函数系数
A = [[1, 1], [-1, 2], [2, 1]]  # 不等式约束矩阵
b = [2, 2, 3]  # 不等式约束向量
bounds = [(0, None), (0, None)]  # 变量界限

result = linprog(c, A_ub=A, b_ub=b, bounds=bounds)
print('最优解：', result.x)
print('目标函数值：', result.fun)
