import numpy as np
import matplotlib.pyplot as plt

class ParticleSwarmOptimization:
    """
    粒子群优化算法类(三维)
    
    属性:
        weight: 浮动,惯性权重,默认值为0.7
        learning_rates: 元组,个体学习率和社会学习率,默认值为(1.5, 1.5)
        max_gen: 整数,最大迭代次数,默认值为3000
        pop_size: 整数,种群大小,默认值为50
        pop_range: 元组,粒子位置范围,默认值为(-2*np.pi, 2*np.pi)
        pop_speed_range: 元组,粒子速度范围,默认值为(-0.5, 0.5)
    """

    def __init__(self, weight=0.7, learning_rates=(1.5, 1.5), max_gen=3000, 
                 pop_size=50, pop_range=(-2*np.pi, 2*np.pi), 
                 pop_speed_range=(-0.5, 0.5)):
        """
        初始化粒子群优化算法
        
        参数:
            weight: 浮动,惯性权重,默认值为0.7
            learning_rates: 元组,个体学习率和社会学习率,默认值为(1.5, 1.5)
            max_gen: 整数,最大迭代次数,默认值为3000
            pop_size: 整数,种群大小,默认值为50
            pop_range: 元组,粒子位置范围,默认值为(-2*np.pi, 2*np.pi)
            pop_speed_range: 元组,粒子速度范围,默认值为(-0.5, 0.5)
        """
        self.weight = weight
        self.learning_rates = learning_rates
        self.max_gen = max_gen
        self.pop_size = pop_size
        self.pop_range = pop_range
        self.pop_speed_range = pop_speed_range
        
        self.pop_positions, self.pop_velocities, self.pop_fitness = self._init_pop_v_fit()
        
        self.pop_gbest, self.pop_gbest_fitness, self.pop_pbest, self.pop_pbest_fitness = self._get_init_best()
        
        self.result = np.zeros(max_gen)

    def _func(self, x):
        """
        计算目标函数的值
        
        参数:
            x: ndarray,输入向量（形状为 (3,)）
        
        返回:
            y: 浮动,目标函数值
        """
        r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
        if r == 0:
            y = (np.exp((np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]) + np.cos(2*np.pi*x[2])) / 3) - 2.71289)
        else:
            y = (np.sin(r) / r + 
                 np.exp((np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]) + np.cos(2*np.pi*x[2])) / 3 - 2.71289))
        return y


    def _init_pop_v_fit(self):
        """
        初始化种群位置、速度和适应度
        
        返回:
            pop_positions: ndarray,种群位置,形状为 (pop_size, 3)
            pop_velocities: ndarray,种群速度,形状为 (pop_size, 3)
            pop_fitness: ndarray,种群适应度,形状为 (pop_size,)
        """
        pop_positions = np.random.uniform(self.pop_range[0], self.pop_range[1], (self.pop_size, 3))
        pop_velocities = np.random.uniform(self.pop_speed_range[0], self.pop_speed_range[1], (self.pop_size, 3))
        pop_fitness = np.apply_along_axis(self._func, 1, pop_positions)
        return pop_positions, pop_velocities, pop_fitness

    def _get_init_best(self):
        """
        获取初始的全局最优和个体最优
        
        返回:
            pop_gbest: ndarray,全局最优位置,形状为 (3,)
            pop_gbest_fitness: 浮动,全局最优适应度
            pop_pbest: ndarray,个体最优位置,形状为 (pop_size, 3)
            pop_pbest_fitness: ndarray,个体最优适应度,形状为 (pop_size,)
        """
        pop_gbest_idx = self.pop_fitness.argmax()
        pop_gbest = self.pop_positions[pop_gbest_idx].copy()
        pop_gbest_fitness = self.pop_fitness[pop_gbest_idx]
        pop_pbest = self.pop_positions.copy()
        pop_pbest_fitness = self.pop_fitness.copy()
        return pop_gbest, pop_gbest_fitness, pop_pbest, pop_pbest_fitness

    def _update_velocities(self, r1, r2):
        """
        更新粒子速度
        
        参数:
            r1: ndarray,随机数,形状为 (pop_size, 3)
            r2: ndarray,随机数,形状为 (pop_size, 3)
        """
        self.pop_velocities = (self.weight * self.pop_velocities +
                               self.learning_rates[0] * r1 * (self.pop_pbest - self.pop_positions) +
                               self.learning_rates[1] * r2 * (self.pop_gbest - self.pop_positions))
        self.pop_velocities = np.clip(self.pop_velocities, self.pop_speed_range[0], self.pop_speed_range[1])

    def _update_positions(self):
        """更新粒子位置"""
        self.pop_positions = self.pop_positions + self.pop_velocities
        self.pop_positions = np.clip(self.pop_positions, self.pop_range[0], self.pop_range[1])

    def _update_fitness(self):
        """更新适应度和最优解"""
        self.pop_fitness = np.apply_along_axis(self._func, 1, self.pop_positions)
        better_pbest_mask = self.pop_fitness > self.pop_pbest_fitness
        self.pop_pbest[better_pbest_mask] = self.pop_positions[better_pbest_mask]
        self.pop_pbest_fitness[better_pbest_mask] = self.pop_fitness[better_pbest_mask]
        if self.pop_pbest_fitness.max() > self.pop_gbest_fitness:
            self.pop_gbest_fitness = self.pop_pbest_fitness.max()
            self.pop_gbest = self.pop_positions[self.pop_pbest_fitness.argmax()].copy()

    def print_all_positions(self, title):
        """
        打印所有粒子的位置
        
        参数:
            title: 字符串,输出的标题
        """
        print("\n" + "="*50)
        print(f"{title}")
        print("\n粒子位置:")
        for i, pos in enumerate(self.pop_positions):
            print(f"粒子 {i+1:2d}: X = {pos[0]:8.4f}, Y = {pos[1]:8.4f}, Z = {pos[2]:8.4f}")

    def print_global_best(self, title):
        """
        打印全局最优位置和适应度
        
        参数:
            title: 字符串,输出的标题
        """
        print("\n")
        print(f"全局最优位置: {self.pop_gbest}")
        print(f"全局最优适应度: {self.pop_gbest_fitness}")
        print("="*50 + "\n")
    
    def plot_state(self, title):
        """
        绘制当前状态
        
        参数:
            title: 字符串,图形标题
        """
        # 设置全局作图样式
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        plt.style.use('seaborn-v0_8-deep')  # 调用 Matplotlib 样式风格,推荐seaborn-v0_8-deep 或 ggplot
        
        # =============================================================================
        # 可用的 Matplotlib 样式列表
        # =============================================================================
        # 'Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid',
        # 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale',
        # 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark',
        # 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted',
        # 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk',
        # 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10'
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1]})
        
        # 左图：粒子位置
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f'{title} - 粒子位置')   
    
        # 绘制所有粒子
        scatter = ax1.scatter(self.pop_positions[:, 0], self.pop_positions[:, 1], self.pop_positions[:, 2],
                              c=np.linalg.norm(self.pop_positions, axis=1), cmap='viridis', alpha=0.8, s=50, label='粒子')
    
        # 突出显示全局最优位置
        ax1.scatter(self.pop_gbest[0], self.pop_gbest[1], self.pop_gbest[2], c='red', 
                    s=100, label='全局最优')
    
        # 添加色条
        cbar = plt.colorbar(scatter, ax=ax1, shrink=0.5)
        cbar.set_label('粒子位置的范数')
    
        ax1.legend()
    
        # 右图：最佳适应度
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('最佳适应度')
        ax2.set_title(f'{title} - 最佳适应度')
    
        if title == "初始状态":
            # 只绘制初始状态下的最佳适应度
            ax2.scatter(0, self.pop_gbest_fitness, color='teal', s=100, label='初始全局最优适应度')
            ax2.set_xlim(-1, 1)  # 设置x轴范围
            ax2.set_ylim(self.pop_gbest_fitness - 1, self.pop_gbest_fitness + 1)  # 设置y轴范围,避免文字重叠
        else:
            # 绘制所有迭代的最佳适应度
            ax2.plot(self.result, color='teal', linewidth=2, marker='o', markersize=5, label='最佳适应度')
            ax2.set_xlim(0, self.max_gen - 1)  # 设置x轴范围
            ax2.set_ylim(self.result.min() - 1, self.result.max() + 1)  # 设置y轴范围,避免文字重叠
    
        ax2.legend()
        plt.tight_layout()
        plt.show()


    def update(self):
        """执行粒子群优化算法"""
        # 记录初始最佳适应度
        self.result[0] = self.pop_gbest_fitness
        for i in range(1, self.max_gen):  # 从1开始,因为0已经用于初始状态
            r1, r2 = np.random.rand(2, self.pop_size, 3)
            self._update_velocities(r1, r2)
            self._update_positions()
            self._update_fitness()
            self.result[i] = self.pop_gbest_fitness


# 使用示例
if __name__ == "__main__":
    pso = ParticleSwarmOptimization()
    pso.print_all_positions("初始状态")
    pso.print_global_best("初始状态")
    pso.plot_state("初始状态")

    pso.update()

    pso.print_all_positions("最终状态")
    pso.print_global_best("最终状态")
    pso.plot_state("最终状态")
