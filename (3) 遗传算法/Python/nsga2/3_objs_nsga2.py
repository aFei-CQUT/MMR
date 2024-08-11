# 三个函数的 nsga2 算法优化
from problem import Problem
from evolution import Evolution
import matplotlib.pyplot as plt

# 目标函数定义
def f1(x):
    return x ** 2

def f2(x):
    return (x - 2) ** 2

def f3(x):
    return (x + 1) ** 2

# 问题定义和求解
problem = Problem(num_of_variables=1, objectives=[f1, f2, f3], variables_range=[(-55, 55)])
evo = Evolution(problem)
evol = evo.evolve()
func = [i.objectives for i in evol]

# 提取目标函数值
function1 = [i[0] for i in func]
function2 = [i[1] for i in func]
function3 = [i[2] for i in func]

# 计算总适应度
total_fitness = [f1_val + f2_val + f3_val for f1_val, f2_val, f3_val in zip(function1, function2, function3)]

# 设置全局样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
plt.style.use('seaborn-v0_8-deep')            # 使用指定样式风格

# 创建图形
fig = plt.figure(figsize=(10, 8))

# 创建 3D 图形
ax = fig.add_subplot(111, projection='3d')

# 绘制目标函数值
scatter = ax.scatter(function1, function2, function3, c=total_fitness, cmap='viridis', alpha=0.7, edgecolors='k')

# 设置标题和标签
ax.set_title(r'Pareto Front of Multi-Objective Optimization', fontsize=16, fontweight='bold')
ax.set_xlabel(r'Function 1', fontsize=14, fontweight='bold')
ax.set_ylabel(r'Function 2', fontsize=14, fontweight='bold')
ax.set_zlabel(r'Function 3', fontsize=14, fontweight='bold')

# 添加网格
ax.grid(True, linestyle='--', alpha=0.7)

# 添加色条
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Total Fitness Value', fontsize=12)

# 显示图形
plt.tight_layout()
plt.show()