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
