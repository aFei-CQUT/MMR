import warnings
import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2  # Import NSGA-II
from pymoo.optimize import minimize

# 忽略RuntimeWarning
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# 定义多维多目标优化问题
class MyProblemND(Problem):
    def __init__(self, n_var):
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=np.zeros(n_var), xu=np.ones(n_var))
        # 初始检查
        self._check_initial_conditions()

    def _check_initial_conditions(self):
        initial_points = np.array([[0.0], [1.0]])  # 示例初始点
        f1 = np.sum(np.power(initial_points, 2), axis=1)
        f2 = np.sum(np.power(initial_points - 1, 2), axis=1)
        print("Initial f1:", f1)
        print("Initial f2:", f2)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = np.sum(np.power(x, 2), axis=1)  # 目标函数1
        f2 = np.sum(np.power(x - 1, 2), axis=1)  # 目标函数2

        if np.any(np.isnan(f1)) or np.any(np.isinf(f1)) or np.any(np.isnan(f2)) or np.any(np.isinf(f2)):
            print("NaN or Inf detected in objectives")
            out["F"] = np.full((x.shape[0], self.n_obj), np.nan)
        else:
            out["F"] = np.column_stack([f1, f2])

# 定义问题维度
n_var = 3  # 可以根据需要修改维度

# 创建问题实例
problem = MyProblemND(n_var)

# 设置NSGA-II算法
algorithm = NSGA2(pop_size=50)

# 优化
res = minimize(problem,
               algorithm,
               termination=('n_gen', 200),
               seed=1,
               verbose=True)

# 处理优化结果中的无效值
def clean_results(res):
    f_values = res.F
    valid_indices = ~np.isnan(f_values).any(axis=1) & ~np.isinf(f_values).any(axis=1)
    valid_f_values = f_values[valid_indices]
    return valid_f_values

# 获取清理后的结果
valid_f_values = clean_results(res)

# 检查有效结果
print("Valid Pareto Front:", valid_f_values)

# 计算总适应度
total_fitness = np.sum(valid_f_values, axis=1)

# 设置全局样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.style.use('seaborn-v0_8-deep')  # 使用 Seaborn 样式风格
plt.rcParams['text.usetex'] = True  # 启用 LaTeX 渲染

# 创建子图
fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # 比例 2:1

# 可视化有效结果（前沿图）
ax1 = axs[0]
sc = ax1.scatter(valid_f_values[:, 0], valid_f_values[:, 1], c=total_fitness, cmap='viridis', label="Pareto Front", edgecolor='k')
ax1.set_title(r'Pareto Front of Multi-Objective Optimization', fontsize=16, fontweight='bold')
ax1.set_xlabel(r'$f_1(x)$', fontsize=14, fontweight='bold')
ax1.set_ylabel(r'$f_2(x)$', fontsize=14, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# 添加色条，表示总适应度
cbar = plt.colorbar(sc, ax=ax1, orientation='vertical')
cbar.set_label(r'Total Fitness Value', fontsize=14, fontweight='bold')
cbar.ax.tick_params(labelsize=12)

# 适应度随迭代次数变化图
history = res.history
best_fitness = [h.pop.get("F").min(axis=0) for h in history]

# 迭代次数
iterations = range(len(best_fitness))

ax2 = axs[1]
if best_fitness:  # Check if best_fitness is not empty
    for i in range(best_fitness[0].shape[0]):
        # Plot each objective's best fitness over iterations with a label
        ax2.plot(iterations, [bf[i] for bf in best_fitness], label=f'Objective {i+1}')
    
    # Add the legend only if there are lines to show
    ax2.legend()
else:
    print("No best fitness values found.")

ax2.set_title('Best Fitness Over Iterations', fontsize=16, fontweight='bold')
ax2.set_xlabel('Iterations', fontsize=14, fontweight='bold')
ax2.set_ylabel('Fitness Value', fontsize=14, fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()