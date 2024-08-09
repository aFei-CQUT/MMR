from nsga2.problem import Problem
from nsga2.evolution import Evolution
import matplotlib.pyplot as plt
import math


def f1(x):
    s = 0
    for i in range(len(x) - 1):
        s += -10 * math.exp(-0.2 * math.sqrt(x[i] ** 2 + x[i + 1] ** 2))
    return s


def f2(x):
    s = 0
    for i in range(len(x)):
        s += abs(x[i]) ** 0.8 + 5 * math.sin(x[i] ** 3)
    return s


problem = Problem(num_of_variables=2, objectives=[f1, f2], variables_range=[(-5, 5)], same_range=True, expand=False)
evo = Evolution(problem, mutation_param=20)
func = [i.objectives for i in evo.evolve()]

function1 = [i[0] for i in func]
function2 = [i[1] for i in func]

# 设置全局样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.style.use('seaborn-v0_8-deep')  # 使用 Seaborn 样式风格

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制目标函数值
scatter = ax.scatter(function1, function2, c=function2, cmap='viridis', alpha=0.7, edgecolors='k')

# 添加色条
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label(r'Function 2 Value', fontsize=12)

# 设置标题和标签
ax.set_title(r'Pareto Front of Multi-Objective Optimization', fontsize=16, fontweight='bold')
ax.set_xlabel(r'Function 1', fontsize=14, fontweight='bold')
ax.set_ylabel(r'Function 2', fontsize=14, fontweight='bold')

# 添加网格
ax.grid(True, linestyle='--', alpha=0.7)

# 显示图形
plt.tight_layout()
plt.show()
