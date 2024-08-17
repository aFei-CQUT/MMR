import numpy as np
import matplotlib.pyplot as plt

class MultiObjectiveParticleSwarmOptimization:
    """
    三目标粒子群优化算法类

    属性:
        weight: 浮动，惯性权重，默认值为0.7
        learning_rates: 元组，个体学习率和社会学习率，默认值为(1.5, 1.5)
        max_gen: 整数，最大迭代次数，默认值为3000
        pop_size: 整数，种群大小，默认值为50
        pop_space_range: 元组，粒子位置范围，默认值为(-2*np.pi, 2*np.pi)
        pop_speed_range: 元组，粒子速度范围，默认值为(-0.5, 0.5)
        objectives: 列表，目标函数列表，默认值为空
        fitness_functions: 列表，适应度计算函数列表，默认值为空
    """

    def __init__(self, weight=0.7, learning_rates=(1.5, 1.5), max_gen=3000, 
                 pop_size=50, pop_space_range=(-2*np.pi, 2*np.pi), 
                 pop_speed_range=(-0.5, 0.5), objectives=None, fitness_functions=None):
        """
        初始化粒子群优化算法
        
        参数:
            weight: 浮动，惯性权重，默认值为0.7
            learning_rates: 元组，个体学习率和社会学习率，默认值为(1.5, 1.5)
            max_gen: 整数，最大迭代次数，默认值为3000
            pop_size: 整数，种群大小，默认值为50
            pop_space_range: 元组，粒子位置范围，默认值为(-2*np.pi, 2*np.pi)
            pop_speed_range: 元组，粒子速度范围，默认值为(-0.5, 0.5)
            objectives: 列表，目标函数列表，默认值为空
            fitness_functions: 列表，适应度计算函数列表，默认值为空
        """
        self.weight = weight
        self.learning_rates = learning_rates
        self.max_gen = max_gen
        self.pop_size = pop_size
        self.pop_space_range = pop_space_range
        self.pop_speed_range = pop_speed_range
        self.objectives = objectives if objectives is not None else [self._objective1, self._objective2, self._objective3]
        self.fitness_functions = fitness_functions if fitness_functions is not None else [self._default_fitness]
        
        self.n_dim = 2  # 粒子维度
        self.n_objectives = len(self.objectives)
        
        self.pop_positions, self.pop_velocities, self.pop_fitness = self._init_pop_v_fit()
        self.pop_gbest, self.pop_gbest_fitness, self.pop_pbest, self.pop_pbest_fitness = self._get_init_best()
        self.result = np.zeros(max_gen)

    def _objective1(self, x):
        """
        第一个目标函数
        """
        return np.sum(x**2)  # 示例目标函数

    def _objective2(self, x):
        """
        第二个目标函数
        """
        return np.sum((x - 1)**2)  # 示例目标函数

    def _objective3(self, x):
        """
        第三个目标函数
        """
        return np.sum((x + 1)**2)  # 示例目标函数

    def _default_fitness(self, values):
        """
        默认适应度计算方法：对于每个目标函数，计算其适应度
        """
        epsilon = 1e-6
        fitness = np.array([1 / (value + epsilon) for value in values.T])
        return fitness.T

    def _calculate_fitness(self, values):
        """
        计算适应度值
        
        参数:
            values: ndarray，目标函数值
        
        返回:
            fitness: ndarray，适应度值
        """
        fitness = np.zeros_like(values)
        for i, func in enumerate(self.fitness_functions):
            fitness[:, i] = func(values[:, i])
        return fitness

    def _init_pop_v_fit(self):
        """
        初始化种群位置、速度和适应度
        
        返回:
            pop_positions: ndarray，种群位置，形状为 (pop_size, n_dim)
            pop_velocities: ndarray，种群速度，形状为 (pop_size, n_dim)
            pop_fitness: ndarray，种群适应度，形状为 (pop_size, n_objectives)
        """
        pop_positions = np.random.uniform(self.pop_space_range[0], self.pop_space_range[1], (self.pop_size, self.n_dim))
        pop_velocities = np.random.uniform(self.pop_speed_range[0], self.pop_speed_range[1], (self.pop_size, self.n_dim))
        
        # 计算每个粒子的适应度
        fitness_values = np.array([np.array([obj(pos) for obj in self.objectives]) for pos in pop_positions])
        pop_fitness = self._calculate_fitness(fitness_values)
        
        return pop_positions, pop_velocities, pop_fitness

    def _get_init_best(self):
        """
        获取初始的全局最优和个体最优
        
        返回:
            pop_gbest: ndarray，全局最优位置，形状为 (n_dim,)
            pop_gbest_fitness: ndarray，全局最优适应度，形状为 (n_objectives,)
            pop_pbest: ndarray，个体最优位置，形状为 (pop_size, n_dim)
            pop_pbest_fitness: ndarray，个体最优适应度，形状为 (pop_size, n_objectives)
        """
        pop_gbest_idx = np.argmin(self.pop_fitness.min(axis=1))  # 选择最小化最优解
        
        pop_gbest = self.pop_positions[pop_gbest_idx].copy()
        pop_gbest_fitness = self.pop_fitness[pop_gbest_idx]
        pop_pbest = self.pop_positions.copy()
        pop_pbest_fitness = self.pop_fitness.copy()
        
        return pop_gbest, pop_gbest_fitness, pop_pbest, pop_pbest_fitness

    def _update_velocities(self, r1, r2):
        """
        更新粒子速度
        
        参数:
            r1: ndarray，随机数，形状为 (pop_size, n_dim)
            r2: ndarray，随机数，形状为 (pop_size, n_dim)
        """
        self.pop_velocities = (self.weight * self.pop_velocities +
                               self.learning_rates[0] * r1 * (self.pop_pbest - self.pop_positions) +
                               self.learning_rates[1] * r2 * (self.pop_gbest - self.pop_positions))
        self.pop_velocities = np.clip(self.pop_velocities, self.pop_speed_range[0], self.pop_speed_range[1])

    def _update_positions(self):
        """更新粒子位置"""
        self.pop_positions = self.pop_positions + self.pop_velocities
        self.pop_positions = np.clip(self.pop_positions, self.pop_space_range[0], self.pop_space_range[1])

    def _update_fitness(self):
        """更新适应度和最优解"""
        new_fitness = np.array([np.array([obj(pos) for obj in self.objectives]) for pos in self.pop_positions])
        new_fitness = self._calculate_fitness(new_fitness)
        
        # 更新个体最优
        better_pbest_mask = np.any(new_fitness < self.pop_pbest_fitness, axis=1)
        self.pop_pbest[better_pbest_mask] = self.pop_positions[better_pbest_mask]
        self.pop_pbest_fitness[better_pbest_mask] = new_fitness[better_pbest_mask]
        
        # 更新全局最优
        if np.any(new_fitness.min(axis=0) < self.pop_gbest_fitness.min(axis=0)):
            self.pop_gbest_fitness = new_fitness.min(axis=0)
            self.pop_gbest = self.pop_positions[np.argmin(new_fitness.min(axis=1))].copy()

    def print_all_positions(self, title):
        """
        打印所有粒子的位置
        
        参数:
            title: 字符串，输出的标题
        """
        print("\n" + "="*50)
        print(f"{title}")
        print("\n粒子位置:")
        for i, pos in enumerate(self.pop_positions):
            print(f"粒子 {i+1:2d}: X = {pos[0]:8.4f}, Y = {pos[1]:8.4f}")

    def print_global_best(self, title):
        """
        打印全局最优位置和适应度
        
        参数:
            title: 字符串，输出的标题
        """
        print("\n")
        print(f"全局最优位置: {self.pop_gbest}")
        print(f"全局最优适应度: {self.pop_gbest_fitness}")
        print("="*50 + "\n")
    
    def plot_state(self, title):
        """
        绘制当前状态并美化图形
        
        参数:
            title: 字符串，图形标题
        """
        fig = plt.figure(figsize=(12, 8))  # 调整图形大小以获得更好的纵横比
        
        # 设置全局样式
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        plt.style.use('seaborn-v0_8-deep')  # 调用 Matplotlib 样式风格
        
        # 使用 gridspec 调整子图比例
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])  # 2:1的比例

        # 3D 图：粒子适应度分布
        ax = fig.add_subplot(gs[0], projection='3d')
        ax.set_xlabel('Objective 1 Fitness', fontsize=12, fontweight='bold')
        ax.set_ylabel('Objective 2 Fitness', fontsize=12, fontweight='bold')
        ax.set_zlabel('Objective 3 Fitness', fontsize=12, fontweight='bold')
        ax.set_title(f'{title} - 粒子群适应度分布图', fontsize=14, fontweight='bold', pad=56)  # 调整标题位置高度

        # 计算颜色映射
        norm = plt.Normalize(self.pop_fitness.min(), self.pop_fitness.max())
        cmap = plt.get_cmap('viridis')

        # 绘制所有粒子的适应度
        sc = ax.scatter(self.pop_fitness[:, 0], self.pop_fitness[:, 1], self.pop_fitness[:, 2], 
                        c=np.sum(self.pop_fitness, axis=1), cmap=cmap, norm=norm, alpha=0.6, s=50, edgecolors='k', label='Particles')

        # 突出显示全局最优位置
        ax.scatter(self.pop_gbest_fitness[0], self.pop_gbest_fitness[1], self.pop_gbest_fitness[2], c='r', s=100, label='Global Best')

        # 添加色条
        cbar = plt.colorbar(sc, ax=ax)  # 添加色条
        cbar.set_label('Total Fitness', fontsize=12)  # 设置色条标签
        cbar.ax.tick_params(width=0.5)  # 设置色条刻度宽度

        # 设置色条的宽度
        cbar.ax.set_aspect(20)  # 调整色条的宽高比，值越大，色条越宽

        ax.legend()

        # 设置 spines
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)  # 调整 spine 的宽度

        # 最佳适应度图
        ax2 = fig.add_subplot(gs[1])
        ax2.set_xlabel('Iterations', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Best Fitness', fontsize=12, fontweight='bold')
        ax2.set_title(f'{title} - 最佳适应度', fontsize=14, fontweight='bold', pad=20)  # 调整标题位置高度

        if title == "Initial State":
            ax2.plot([0], [self.pop_gbest_fitness.min()], 'ro', markersize=8)
        else:
            ax2.plot(self.result, color='darkorange', linewidth=2, marker='o', markersize=6)

        ax2.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()  # 自动调整子图参数
        plt.show()  # 显示图形

    def update(self):
        """执行粒子群优化算法"""
        # 记录初始最佳适应度
        self.result[0] = self.pop_gbest_fitness.min()
        for i in range(1, self.max_gen):  # 从1开始，因为0已经用于初始状态
            r1, r2 = np.random.rand(2, self.pop_size, self.n_dim)
            self._update_velocities(r1, r2)
            self._update_positions()
            self._update_fitness()
            self.result[i] = self.pop_gbest_fitness.min()

# 使用示例
if __name__ == "__main__":
# 适应度计算方法示例
    def fitness1(values):
        return 1 / (values + 1e-6)
    
    def fitness2(values):
        return np.sqrt(values + 1e-6)
    
    def fitness3(values):
        return 1 / (values + 1e-6)
    
    pso = MultiObjectiveParticleSwarmOptimization(
        objectives=[lambda x: np.sum(x**2), lambda x: np.sum((x - 1)**2), lambda x: np.sum((x + 1)**2)],
        fitness_functions=[fitness1, fitness2, fitness3]
    )
    pso.print_all_positions("初始状态")
    pso.print_global_best("初始状态")
    pso.plot_state("初始状态")
    
    pso.update()
    
    pso.print_all_positions("最终状态")
    pso.print_global_best("最终状态")
    pso.plot_state("最终状态")