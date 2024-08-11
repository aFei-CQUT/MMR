import matplotlib.pyplot as plt
from problem import Problem
from evolution import Evolution

# 目标函数定义
def f1(x):
    return x ** 2

def f2(x):
    return (x - 2) ** 2

# 问题定义和求解
problem = Problem(num_of_variables=1, objectives=[f1, f2], variables_range=[(-55, 55)])
evo = Evolution(problem)
evol = evo.evolve()
func = [i.objectives for i in evol]

# 提取目标函数值
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
cbar.set_label('Function 2 Value', fontsize=12)

# 设置标题和标签
ax.set_title(r'Pareto Front of Multi-Objective Optimization', fontsize=16, fontweight='bold')
ax.set_xlabel(r'Function 1', fontsize=14, fontweight='bold')
ax.set_ylabel(r'Function 2', fontsize=14, fontweight='bold')

# 添加网格
ax.grid(True, linestyle='--', alpha=0.7)

# 显示图形
plt.tight_layout()
plt.show()
