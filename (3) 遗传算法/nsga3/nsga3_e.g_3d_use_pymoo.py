import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize

# 忽略RuntimeWarning
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# 定义三维多目标优化问题
class MyProblem3D(Problem):
    def __init__(self):
        super().__init__(n_var=1, n_obj=3, n_constr=0, xl=np.array([0.0]), xu=np.array([1.0]))
        # 初始检查
        self._check_initial_conditions()

    def _check_initial_conditions(self):
        initial_points = np.array([[0.0], [1.0]])  # 示例初始点
        f1 = initial_points[:, 0]**2
        f2 = (initial_points[:, 0] - 1)**2
        f3 = (initial_points[:, 0] - 0.5)**2
        print("Initial f1:", f1)
        print("Initial f2:", f2)
        print("Initial f3:", f3)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = np.power(x[:, 0], 2)
        f2 = np.power(x[:, 0] - 1, 2)
        f3 = np.power(x[:, 0] - 0.5, 2)

        if np.any(np.isnan(f1)) or np.any(np.isinf(f1)) or np.any(np.isnan(f2)) or np.any(np.isinf(f2)) or np.any(np.isnan(f3)) or np.any(np.isinf(f3)):
            print("NaN or Inf detected in objectives")
            out["F"] = np.full((x.shape[0], self.n_obj), np.nan)
        else:
            out["F"] = np.column_stack([f1, f2, f3])

problem = MyProblem3D()

# 定义参考方向
ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

# 设置NSGA-III算法
algorithm = NSGA3(pop_size=50, ref_dirs=ref_dirs)

# 优化
res = minimize(problem,
               algorithm,
               termination=('n_gen', 200),
               seed=1,
               verbose=True,
               save_history=True)  # 保存历史记录

# 处理优化结果中的无效值
def clean_results(res):
    f_values = res.F
    valid_indices = ~np.isnan(f_values).any(axis=1) & ~np.isinf(f_values).any(axis=1)
    valid_f_values = f_values[valid_indices]
    return valid_f_values

# 获取清理后的结果
valid_f_values = clean_results(res)

# 设置全局样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.style.use('seaborn-v0_8-deep')  # 使用 Seaborn 样式风格
plt.rcParams['text.usetex'] = True  # 启用 LaTeX 渲染

# 创建子图
fig, axs = plt.subplots(1, 2, figsize=(20, 10))  # 比例 2:1

# Pareto Front 图
ax1 = fig.add_subplot(121, projection='3d')

# 计算颜色映射
norm = Normalize(vmin=np.min(valid_f_values[:, 0]), vmax=np.max(valid_f_values[:, 0]))
sc = ax1.scatter(valid_f_values[:, 0], valid_f_values[:, 1], valid_f_values[:, 2], c=valid_f_values[:, 0], cmap='viridis', norm=norm)

# 添加色条
cbar = plt.colorbar(sc, ax=ax1, pad=0.1)
cbar.set_label(r'$f_1(x)$', fontsize=14, fontweight='bold')

ax1.set_title(r'Pareto Front of Multi-Objective Optimization', fontsize=16, fontweight='bold')
ax1.set_xlabel(r'$f_1(x)$', fontsize=14, fontweight='bold')
ax1.set_ylabel(r'$f_2(x)$', fontsize=14, fontweight='bold')
ax1.set_zlabel(r'$f_3(x)$', fontsize=14, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.7)

# 适应度随迭代次数变化图
history = res.history
best_fitness = [h.pop.get("F").min(axis=0) for h in history]

# 迭代次数
iterations = range(len(best_fitness))

ax2 = fig.add_subplot(122)
for i in range(best_fitness[0].shape[0]):
    ax2.plot(iterations, [bf[i] for bf in best_fitness], label=f'Objective {i+1}')

ax2.set_title('Best Fitness Over Iterations', fontsize=16, fontweight='bold')
ax2.set_xlabel('Iterations', fontsize=14, fontweight='bold')
ax2.set_ylabel('Fitness Value', fontsize=14, fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

plt.tight_layout()
plt.show()
