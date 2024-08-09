import pygad

# 定义多目标函数
def multiobjective(x):
    return [x[0] ** 2 + x[1] ** 2, (x[0] - 1) ** 2 + (x[1] - 1) ** 2]

# 创建适应度函数
def fitness_func(ga_instance, solution, solution_idx):
    objectives = multiobjective(solution)
    # 通过将目标值的和作为适应度值
    # 这里是一个示例，可以根据具体需要调整
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
