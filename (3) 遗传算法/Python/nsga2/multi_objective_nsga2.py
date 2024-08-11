import random
import matplotlib.pyplot as plt
from tqdm import tqdm

class MultiObjectiveNSGA2:
    def __init__(self, objectives, number_of_variables, variables_range, num_of_generations=1000,
                 num_of_individuals=100, num_of_tour_particips=2, tournament_prob=0.9,
                 crossover_param=2, mutation_param=5, log_file='evolution_log.txt'):
        self.num_of_objectives = len(objectives)  # 目标函数的数量
        self.number_of_variables = number_of_variables  # 决策变量的数量
        self.objectives = objectives  # 目标函数列表
        self.variables_range = (
            [variables_range[0]] * number_of_variables
            if len(variables_range) == 1
            else variables_range
        )  # 变量范围
        self.num_of_generations = num_of_generations  # 迭代次数
        self.num_of_individuals = num_of_individuals  # 种群大小
        self.num_of_tour_particips = num_of_tour_particips  # 锦标赛中参与者的数量
        self.tournament_prob = tournament_prob  # 锦标赛选择概率
        self.crossover_param = crossover_param  # 交叉参数
        self.mutation_param = mutation_param  # 变异参数
        self.population = None  # 当前种群
        self.log_file = log_file  # 输出日志文件
        self.log_data = []  # 存储所有日志数据

    def _generate_individual(self):
        """生成一个个体，包含特征、目标值、等级、拥挤距离等信息"""
        individual = {
            'features': [random.uniform(*x) for x in self.variables_range],
            'objectives': None,
            'rank': None,
            'crowding_distance': None,
            'domination_count': None,
            'dominated_solutions': None
        }
        return individual

    def _calculate_objectives(self, individual):
        """计算个体的目标值"""
        individual['objectives'] = [f(individual['features']) for f in self.objectives]

    def _create_initial_population(self):
        """创建初始种群"""
        population = []
        for _ in range(self.num_of_individuals):
            individual = self._generate_individual()
            self._calculate_objectives(individual)
            population.append(individual)
        return population

    def get_pareto_front(self):
        """获取当前帕累托前沿"""
        if self.population and len(self.population[0].get('fronts', [])) > 0:
            pareto_front = self.population[0]['fronts'][0]
            return [individual['objectives'] for individual in pareto_front]
        else:
            raise ValueError("No Pareto front available. Run the algorithm first.")

    def get_best_solution(self):
        """获取当前的最优解"""
        if self.population and len(self.population[0].get('fronts', [])) > 0:
            pareto_front = self.population[0]['fronts'][0]
            # Assuming minimization problem; modify if needed for maximization
            best_individual = min(pareto_front, key=lambda x: x['objectives'])
            return best_individual['features'], best_individual['objectives']
        else:
            raise ValueError("No Pareto front available. Run the algorithm first.")

    def _write_log(self, message):
        """写入日志文件"""
        self.log_data.append(message)  # 存储日志数据到列表中

    def _print_population(self, generation):
        """打印指定代的所有个体及其目标值"""
        output = [f"第{generation}次迭代\n"]
        for i, individual in enumerate(self.population):
            features = individual['features']
            objectives = individual['objectives']
            output.append(f"Individual {i}: Features = {features}, Objectives = {objectives}")
        self._write_log("\n".join(output))

    def _log_generation_summary(self, generation):
        """记录每代的总结，包括最优值和Pareto前沿"""
        if self.population and len(self.population) > 0 and 'fronts' in self.population[0]:
            if len(self.population[0]['fronts']) > 0:
                # 记录最优值
                pareto_front = self.population[0]['fronts'][0]
                best_individual = min(pareto_front, key=lambda x: x['objectives'])
                best_objectives = best_individual['objectives']
                summary = f"第{generation}次迭代最佳目标值: {best_objectives}\n"
        
                # 记录前沿
                pareto_objectives = [ind['objectives'] for ind in pareto_front]
                summary += "帕累托前沿目标值:\n" + "\n".join(map(str, pareto_objectives)) + "\n"
        
                # 记录前沿的大小
                front_sizes = [len(front) for front in self.population[0]['fronts']]
                summary += "\n".join(f"前沿 {i} 大小: {size}" for i, size in enumerate(front_sizes)) + "\n"
        
                self._write_log(summary)
            else:
                self._write_log(f"第{generation}次迭代 - 没有帕累托前沿可用。\n")
        else:
            self._write_log(f"第{generation}次迭代 - 没有有效的种群或前沿数据。\n")

    def evolve(self):
        """主进化函数，包括初始化、进化和绘图"""
        self.population = self._create_initial_population()
        self._print_population(0)  # 打印初代种群到日志文件
    
        # 确保个体有目标值
        for individual in self.population:
            if individual['objectives'] is None:
                self._calculate_objectives(individual)
    
        # 记录初代的总结信息
        self._log_generation_summary(0)
    
        self._fast_nondominated_sort()
    
        if not self.population or len(self.population[0].get('fronts', [])) == 0:
            self._write_log("错误：快排非支配排序后没有生成前沿。\n")  # 调试：没有生成前沿
    
        for i, front in enumerate(self.population[0]['fronts']):
            self._write_log(f"前沿 {i} 大小: {len(front)}\n")  # 调试：打印每个前沿的大小
            self._calculate_crowding_distance(front)
    
        for generation in tqdm(range(self.num_of_generations)):
            children = self._create_children()
            self.population.extend(children)
            self._fast_nondominated_sort()
            new_population = []
            front_num = 0
            while len(new_population) + len(self.population[0]['fronts'][front_num]) <= self.num_of_individuals:
                self._calculate_crowding_distance(self.population[0]['fronts'][front_num])
                new_population.extend(self.population[0]['fronts'][front_num])
                front_num += 1
            self._calculate_crowding_distance(self.population[0]['fronts'][front_num])
            self.population[0]['fronts'][front_num].sort(key=lambda individual: individual['crowding_distance'], reverse=True)
            new_population.extend(self.population[0]['fronts'][front_num][:self.num_of_individuals - len(new_population)])
            self.population = new_population
    
            # 打印每代的种群
            self._print_population(generation + 1)
            
            # 记录每代的总结信息
            self._log_generation_summary(generation + 1)
    
        # 确保有前沿可以绘图
        if self.population and len(self.population[0]['fronts']) > 0:
            self._plot_pareto()
        else:
            self._write_log("错误：没有可用的Pareto前沿进行绘图。\n")
    
        # 将所有日志数据写入文件
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.log_data))
                
    def _fast_nondominated_sort(self):
        """快速非支配排序"""
        fronts = [[]]
        for individual in self.population:
            individual['domination_count'] = 0
            individual['dominated_solutions'] = []
            for other_individual in self.population:
                if self._dominates(individual, other_individual):
                    individual['dominated_solutions'].append(other_individual)
                elif self._dominates(other_individual, individual):
                    individual['domination_count'] += 1
            if individual['domination_count'] == 0:
                individual['rank'] = 0
                fronts[0].append(individual)
    
        i = 0
        while len(fronts[i]) > 0:
            temp = []
            for individual in fronts[i]:
                for other_individual in individual['dominated_solutions']:
                    other_individual['domination_count'] -= 1
                    if other_individual['domination_count'] == 0:
                        other_individual['rank'] = i + 1
                        temp.append(other_individual)
            i += 1
            fronts.append(temp)
        
        if len(fronts) > 0:
            self._write_log(f"前沿数量: {len(fronts)}\n")  # 记录前沿数量
        self.population[0]['fronts'] = fronts
    
    def _dominates(self, individual1, individual2):
        """判断个体1是否支配个体2"""
        and_condition = True
        or_condition = False
        for f1, f2 in zip(individual1['objectives'], individual2['objectives']):
            and_condition = and_condition and f1 <= f2
            or_condition = or_condition or f1 < f2
        return and_condition and or_condition
    
    def _calculate_crowding_distance(self, front):
        """计算个体的拥挤距离"""
        if len(front) > 0:
            solutions_num = len(front)
            for individual in front:
                individual['crowding_distance'] = 0
            for m in range(self.num_of_objectives):
                front.sort(key=lambda x: x['objectives'][m])
                front[0]['crowding_distance'] = float('inf')
                front[-1]['crowding_distance'] = float('inf')
                for i in range(1, solutions_num - 1):
                    front[i]['crowding_distance'] += (front[i + 1]['objectives'][m] - front[i - 1]
                                                      ['objectives'][m]) / (front[-1]['objectives'][m] - front[0]['objectives'][m])
    
    def _create_children(self):
        """创建下一代的子代"""
        children = []
        while len(children) < self.num_of_individuals:
            parent1 = self._tournament()
            parent2 = self._tournament()
            child1, child2 = self._crossover(parent1, parent2)
            self._mutation(child1)
            self._mutation(child2)
            self._calculate_objectives(child1)
            self._calculate_objectives(child2)
            children.append(child1)
            children.append(child2)
        return children
    
    def _crossover(self, individual1, individual2):
        """交叉操作"""
        child1, child2 = self._generate_individual(), self._generate_individual()
        for i in range(len(individual1['features'])):
            beta = self._get_beta()
            x1 = (individual1['features'][i] + individual2['features'][i]) / 2
            x2 = abs((individual1['features'][i] - individual2['features'][i]) / 2)
            child1['features'][i] = x1 + beta * x2
            child2['features'][i] = x1 - beta * x2
        return child1, child2
    
    def _get_beta(self):
        """计算交叉操作中的beta值"""
        u = random.random()
        if u <= 0.5:
            return (2 * u) ** (1 / (self.crossover_param + 1))
        return (2 * (1 - u)) ** (-1 / (self.crossover_param + 1))
    
    def _mutation(self, child):
        """变异操作"""
        num_of_features = len(child['features'])
        for gene in range(num_of_features):
            u, delta = self._get_delta()
            if u < 0.5:
                child['features'][gene] += delta * (child['features'][gene] - self.variables_range[gene][0])
            else:
                child['features'][gene] += delta * (self.variables_range[gene][1] - child['features'][gene])
            # 确保特征值在变量范围内
            if child['features'][gene] < self.variables_range[gene][0]:
                child['features'][gene] = self.variables_range[gene][0]
            elif child['features'][gene] > self.variables_range[gene][1]:
                child['features'][gene] = self.variables_range[gene][1]
    
    def _get_delta(self):
        """计算变异操作中的delta值"""
        u = random.random()
        if u < 0.5:
            return u, (2 * u) ** (1 / (self.mutation_param + 1)) - 1
        return u, 1 - (2 * (1 - u)) ** (1 / (self.mutation_param + 1))

    def _tournament(self):
        """锦标赛选择操作"""
        # 确保种群大小不小于比赛所需的参与者数量
        num_participants = min(len(self.population), self.num_of_tour_particips)
        participants = random.sample(self.population, num_participants)
        best = None
        for participant in participants:
            if best is None or (
                    self._crowding_operator(participant, best) == 1 and self._choose_with_prob(self.tournament_prob)):
                best = participant
        return best
    
    def _crowding_operator(self, individual1, individual2):
        """根据拥挤距离和等级比较两个个体"""
        if individual1['rank'] < individual2['rank']:
            return 1
        elif individual1['rank'] > individual2['rank']:
            return -1
        elif individual1['crowding_distance'] > individual2['crowding_distance']:
            return 1
        elif individual1['crowding_distance'] < individual2['crowding_distance']:
            return -1
        return 0
    
    def _choose_with_prob(self, prob):
        """根据给定的概率进行选择"""
        return random.random() <= prob
    
    def _plot_pareto(self):
        """绘制种群的Pareto前沿"""
        plt.close('all')  # 关闭所有现有图形，以确保新图形不会与旧图形重叠
        fig = plt.figure(figsize=(12, 8))  # 调整图形大小以获得更好的纵横比
        
        # 设置全局样式
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        plt.style.use('seaborn-v0_8-deep')  # 调用 Matplotlib 样式风格
        
        if not self.population or len(self.population[0].get('fronts', [])) == 0:
            self._write_log("错误：没有可用的Pareto前沿进行绘图。\n")
            return
        
        pareto_front = self.population[0]['fronts'][0]
        fig = plt.figure()
        if len(pareto_front[0]['objectives']) > 2:
            # 如果目标函数大于2，使用3D绘图
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('目标1')
            ax.set_ylabel('目标2')
            ax.set_zlabel('目标3')
            for individual in pareto_front:
                ax.scatter(individual['objectives'][0], individual['objectives'][1],
                           individual['objectives'][2], marker='o')
        elif len(pareto_front[0]['objectives']) == 2:
            # 如果目标函数为2，使用2D绘图
            ax = fig.add_subplot(111)
            ax.set_xlabel('目标1')
            ax.set_ylabel('目标2')
            for individual in pareto_front:
                ax.scatter(individual['objectives'][0], individual['objectives'][1], marker='o')
        else:
            self._write_log("错误：目标函数必须是2或3。\n")
        plt.title('Pareto Front')
        plt.grid(True)
        plt.show()
    
# 主程序部分（无需修改，除非你希望在这里使用新的方法）
if __name__ == '__main__':
    # 示例目标函数
    def objective_1(x):
        return x[0] ** 2 + x[1] ** 2

    def objective_2(x):
        return (x[0] - 1) ** 2 + x[1] ** 2

    def objective_3(x):
        return x[0] ** 2 + (x[1] - 1) ** 2

    # 设置全局作图样式，禁用LaTeX渲染
    plt.rcParams['font.family'] = ['SimHei']  # 使用常用字体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    plt.rcParams['text.usetex'] = False  # 禁用LaTeX渲染
    plt.rcParams['axes.grid'] = True  # 启用网格
    plt.style.use('seaborn-v0_8-deep')  # 使用Seaborn样式
    
    # 定义问题
    number_of_variables = 2
    variables_range = [(0, 1), (0, 1)]  # 每个变量的范围
    objectives = [objective_1, objective_2, objective_3]
    
    # 初始化算法
    num_of_individuals = 5
    num_of_generations = 5
    
    algorithm = MultiObjectiveNSGA2(
        objectives=objectives, 
        number_of_variables=number_of_variables, 
        variables_range=variables_range, 
        num_of_generations=num_of_generations, 
        num_of_individuals=num_of_individuals,
        log_file='evolution_log.txt'  # 指定日志文件路径
    )
    
    # 开始进化
    algorithm.evolve()

    # 获取帕累托前沿和最优解
    try:
        pareto_front = algorithm.get_pareto_front()
        best_solution_features, best_solution_objectives = algorithm.get_best_solution()
        print("Pareto Front:", pareto_front)
        print("Best Solution Features:", best_solution_features)
        print("Best Solution Objectives:", best_solution_objectives)
    except ValueError as e:
        print(e)
