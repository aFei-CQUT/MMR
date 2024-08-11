import warnings
import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize

# 忽略RuntimeWarning
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# 定义一维多目标优化问题
class MyProblem1D(Problem):
    def __init__(self):
        super().__init__(n_var=1, n_obj=2, n_constr=0, xl=np.array([0.0]), xu=np.array([1.0]))
        # 初始检查
        self._check_initial_conditions()

    def _check_initial_conditions(self):
        initial_points = np.array([[0.0], [1.0]])  # 示例初始点
        f1 = initial_points[:, 0]**2
        f2 = (initial_points[:, 0] - 1)**2
        print("Initial f1:", f1)
        print("Initial f2:", f2)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = np.power(x[:, 0], 2)
        f2 = np.power(x[:, 0] - 1, 2)

        if np.any(np.isnan(f1)) or np.any(np.isinf(f1)) or np.any(np.isnan(f2)) or np.any(np.isinf(f2)):
            print("NaN or Inf detected in objectives")
            out["F"] = np.full((x.shape[0], self.n_obj), np.nan)
        else:
            out["F"] = np.column_stack([f1, f2])

problem = MyProblem1D()

# 定义参考方向
ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)

# 设置NSGA-III算法
algorithm = NSGA3(pop_size=50, ref_dirs=ref_dirs)

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

# 设置全局样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.style.use('seaborn-v0_8-deep')  # 使用 Seaborn 样式风格
plt.rcParams['text.usetex'] = True  # 启用 LaTeX 渲染

# 可视化有效结果（前沿图）
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(valid_f_values[:, 0], valid_f_values[:, 1], c=valid_f_values[:, 0], cmap='viridis', label="Pareto Front", edgecolor='k')
ax.set_title(r'Pareto Front of Multi-Objective Optimization', fontsize=16, fontweight='bold')
ax.set_xlabel(r'$f_1(x)$', fontsize=14, fontweight='bold')
ax.set_ylabel(r'$f_2(x)$', fontsize=14, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend()

# 添加色条
cbar = plt.colorbar(sc, ax=ax, orientation='vertical')
cbar.set_label(r'$f_1(x)$ Value', fontsize=14, fontweight='bold')
cbar.ax.tick_params(labelsize=12)

# 可视化目标函数值
x = np.linspace(0, 1, 100)
f1 = x**2
f2 = (x - 1)**2

fig, ax = plt.subplots(1, 2, figsize=(16, 8))

# 目标函数1
c1 = ax[0].plot(x, f1, label=r'$f_1(x)$', color='b')
ax[0].set_title(r'Objective 1', fontsize=16, fontweight='bold')
ax[0].set_xlabel(r'$x$', fontsize=14, fontweight='bold')
ax[0].set_ylabel(r'$f_1(x)$', fontsize=14, fontweight='bold')
ax[0].grid(True, linestyle='--', alpha=0.7)
ax[0].legend()

# 目标函数2
c2 = ax[1].plot(x, f2, label=r'$f_2(x)$', color='r')
ax[1].set_title(r'Objective 2', fontsize=16, fontweight='bold')
ax[1].set_xlabel(r'$x$', fontsize=14, fontweight='bold')
ax[1].set_ylabel(r'$f_2(x)$', fontsize=14, fontweight='bold')
ax[1].grid(True, linestyle='--', alpha=0.7)
ax[1].legend()

plt.tight_layout()
plt.show()
