### fgoalattain
```python
from scipy.optimize import minimize

def multiobjective(x):
    return [(x[0] ** 2 + x[1] ** 2), (x[0] - 1) ** 2 + (x[1] - 1) ** 2]

goals = [0, 0]
weights = [1, 1]

result = minimize(lambda x: sum(w * abs(f - g) for f, g, w in zip(multiobjective(x), goals, weights)),
                  x0=[0.5, 0.5], bounds=[(0, None), (0, None)])
print('最优解：', result.x)
print('目标函数值：', multiobjective(result.x))
```

### fminbnd
```python
from scipy.optimize import minimize_scalar

result = minimize_scalar(lambda x: (x - 2) ** 2 + 1, bounds=(0, 4), method='bounded')
print('最优解：', result.x)
print('目标函数值：', result.fun)
```

### fminsearch
```python
from scipy.optimize import minimize

result = minimize(lambda x: (x[0] - 2) ** 2 + (x[1] - 3) ** 2, x0=[0, 0], method='Nelder-Mead')
print('最优解：', result.x)
print('目标函数值：', result.fun)
```

### fminunc
```python
from scipy.optimize import minimize

result = minimize(lambda x: (x[0] - 2) ** 2 + (x[1] - 3) ** 2, x0=[0, 0])
print('最优解：', result.x)
print('目标函数值：', result.fun)
```

### ga e.g 1
```python
from deap import base, creator, tools, algorithms
import random

# 创建目标函数
def objective(individual):
    x, y = individual
    return (x - 2) ** 2 + (y - 3) ** 2,

# 创建遗传算法的适应度和个体类型
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 5)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", objective)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# 设置遗传算法的参数
population = toolbox.population(n=50)
algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, verbose=True)

# 输出最优解
best_individual = tools.selBest(population, 1)[0]
print('最优解：', best_individual)
print('目标函数值：', objective(best_individual)[0])
```

### ga e.g 2
```python
import pygad

# 创建适应度函数
def objective_function(ga_instance, solution, solution_idx):
    x, y = solution
    return (x - 2) ** 2 + (y - 3) ** 2,

# 设置遗传算法的参数
num_generations = 50
num_parents_mating = 10
sol_per_pop = 50
num_genes = 2
gene_space = [{'low': 0, 'high': 5}, {'low': 0, 'high': 5}]

# 初始化遗传算法
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=objective_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       gene_space=gene_space,
                       crossover_probability=0.5,
                       mutation_probability=0.2)

# 执行优化
ga_instance.run()

# 输出最优解
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print('最优解：', solution)
print('目标函数值：', solution_fitness)
```

### intlinprog
```python
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
```

### linprog
```python
from scipy.optimize import linprog

c = [-1, -2]  # 目标函数系数
A = [[1, 1], [-1, 2], [2, 1]]  # 不等式约束矩阵
b = [2, 2, 3]  # 不等式约束向量
bounds = [(0, None), (0, None)]  # 变量界限

result = linprog(c, A_ub=A, b_ub=b, bounds=bounds)
print('最优解：', result.x)
print('目标函数值：', result.fun)
```

### paretosearch
```python
import pygad

# 定义多目标函数
def multiobjective(x):
    return [x[0] ** 2 + x[1] ** 2, (x[0] - 1) ** 2 + (x[1] - 1) ** 2]

# 创建适应度函数
def fitness_func(ga_instance, solution, solution_idx):
    objectives = multiobjective(solution)
    return -sum(objectives),

# 设置遗传算法的参数
num_generations = 50
num_parents_mating = 10
sol_per_pop = 50
num_genes = 2
gene_space = [{'low': 0, 'high': 1}, {'low': 0, 'high': 1}]

# 初始化遗传算法
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       gene_space=gene_space,
                       crossover_probability=0.5,
                       mutation_probability=0.2)

# 执行优化
ga_instance.run()

# 获取最优解
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print('最优解：', solution)
print('目标函数值：', multiobjective(solution))
```

### particleswarm
```python
from scipy.optimize import differential_evolution

def objective(x):
    return (x[0] - 2) ** 2 + (x[1] - 3) ** 2

bounds = [(0, 5), (0, 5)]

result = differential_evolution(objective, bounds)
print('最优解：', result.x)
print('目标函数值：', result.fun)
```

### quadprog
```python
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
```

### simulannealbnd
```python
from scipy.optimize import dual_annealing

def objective(x):
    return (x - 2) ** 2 + 1

bounds = [(-5, 5)]



result = dual_annealing(objective, bounds)
print('最优解：', result.x)
print('目标函数值：', result.fun)
```