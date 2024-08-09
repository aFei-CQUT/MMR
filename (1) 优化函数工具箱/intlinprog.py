from pulp import LpProblem, LpMinimize, LpVariable

# 创建优化问题
prob = LpProblem("Integer_Linear_Programming", LpMinimize)

# 定义变量
x1 = LpVariable('x1', lowBound=0, cat='Integer')
x2 = LpVariable('x2', lowBound=0, cat='Integer')

# 目标函数
prob += -1 * x1 - 2 * x2

# 不等式约束
prob += x1 + x2 <= 2
prob += -x1 + 2 * x2 <= 2
prob += 2 * x1 + x2 <= 3

# 求解
prob.solve()

print('最优解：x1 =', x1.varValue, 'x2 =', x2.varValue)
print('目标函数值：', prob.objective.value())
