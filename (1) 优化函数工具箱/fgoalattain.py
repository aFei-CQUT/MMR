from scipy.optimize import minimize

def multiobjective(x):
    return [(x[0] ** 2 + x[1] ** 2), (x[0] - 1) ** 2 + (x[1] - 1) ** 2]

goals = [0, 0]
weights = [1, 1]

result = minimize(lambda x: sum(w * abs(f - g) for f, g, w in zip(multiobjective(x), goals, weights)),
                  x0=[0.5, 0.5], bounds=[(0, None), (0, None)])
print('最优解：', result.x)
print('目标函数值：', multiobjective(result.x))
