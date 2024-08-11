import warnings
import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize

# 忽略RuntimeWarning
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# 定义多目标优化问题
class MyProblem2D(Problem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=0, xl=np.array([0.0, 0.0]), xu=np.array([1.0, 1.0]))
        # 初始检查
        self._check_initial_conditions()

    def _check_initial_conditions(self):
        initial_points = np.array([[0.0, 0.0], [1.0, 1.0]])  # 示例初始点
        f1 = initial_points[:, 0]**2 + initial_points[:, 1]**2
        f2 = (initial_points[:, 0] - 1)**2 + (initial_points[:, 1] - 1)**2
        print("Initial f1:", f1)
        print("Initial f2:", f2)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = np.power(x[:, 0], 2) + np.power(x[:, 1], 2)
        f2 = np.power(x[:, 0] - 1, 2) + np.power(x[:, 1] - 1, 2)

        if np.any(np.isnan(f1)) or np.any(np.isinf(f1)) or np.any(np.isnan(f2)) or np.any(np.isinf(f2)):
            print("NaN or Inf detected in objectives")
            out["F"] = np.full((x.shape[0], self.n_obj), np.nan)
        else:
            out["F"] = np.column_stack([f1, f2])

problem = MyProblem2D()

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

# 设置全局样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.style.use('seaborn-v0_8-deep')  # 使用 Seaborn 样式风格
plt.rcParams['text.usetex'] = True  # 启用 LaTeX 渲染

# 创建子图
fig, ax = plt.subplots(1, 2, figsize=(16, 8))  # 比例 2:1

# Pareto Front 图
sc = ax[0].scatter(valid_f_values[:, 0], valid_f_values[:, 1], c=valid_f_values[:, 0], cmap='viridis', label="Pareto Front")
ax[0].set_title(r'Pareto Front of Multi-Objective Optimization', fontsize=16, fontweight='bold')
ax[0].set_xlabel(r'Objective 1', fontsize=14, fontweight='bold')
ax[0].set_ylabel(r'Objective 2', fontsize=14, fontweight='bold')
ax[0].grid(True, linestyle='--', alpha=0.7)
ax[0].legend()

# 添加色条
cbar = plt.colorbar(sc, ax=ax[0], pad=0.02)
cbar.set_label(r'$Objective 1$', fontsize=14, fontweight='bold')

# 热力图
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z1 = X**2 + Y**2
Z2 = (X - 1)**2 + (Y - 1)**2

c1 = ax[1].contourf(X, Y, Z1, cmap='viridis')
ax[1].set_title(r'Objective 1', fontsize=16, fontweight='bold')
cbar1 = plt.colorbar(c1, ax=ax[1], pad=0.02)
cbar1.set_label(r'$f_1(x)$', fontsize=14, fontweight='bold')

c2 = ax[1].contourf(X, Y, Z2, cmap='viridis', alpha=0.5)
ax[1].set_title(r'Objective 2', fontsize=16, fontweight='bold')
cbar2 = plt.colorbar(c2, ax=ax[1], pad=0.02)
cbar2.set_label(r'$f_2(x)$', fontsize=14, fontweight='bold')

for a in ax:
    a.set_xlabel(r'$x_1$', fontsize=14, fontweight='bold')
    a.set_ylabel(r'$x_2$', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
