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
