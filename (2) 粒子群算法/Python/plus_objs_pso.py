import numpy as np
import matplotlib.pyplot as plt
from pyswarms.single.global_best import GlobalBestPSO
from sklearn.decomposition import PCA
import logging

# 配置日志
logging.getLogger('pyswarms').setLevel(logging.WARNING)

# 定义多个目标函数
def objective_function_1(x):
    return np.sum(100.0 * (x[:, 1:] - x[:, :-1]**2.0)**2.0 + (1 - x[:, :-1])**2.0, axis=1)

def objective_function_2(x):
    return np.sum(np.square(x), axis=1)

def objective_function_3(x):
    return np.sum(np.abs(x), axis=1)

def objective_function_4(x):
    return np.sum(np.sin(x), axis=1)

# 定义每个目标的适应度函数
def fitness_function_1(cost):
    return -cost  # 直接取负值

def fitness_function_2(cost):
    return -cost / np.max(cost)  # 归一化处理

def fitness_function_3(cost):
    return -np.sqrt(cost)  # 开方变换

def fitness_function_4(cost):
    cost_safe = np.clip(cost, 1e-10, None)  # 将0替换为一个小的正值
    return -np.log1p(cost_safe)  # 对数变换

# 设置优化参数
dimensions = 10  # 维度
c1, c2, w = 0.5, 0.3, 0.9  # PSO参数
n_particles = 50  # 粒子数量
iters = 1000  # 迭代次数

# 设置搜索空间边界
bounds = (np.array([-5]*dimensions), np.array([5]*dimensions))

# 初始化优化器
optimizer = GlobalBestPSO(n_particles=n_particles, dimensions=dimensions, 
                          options={'c1': c1, 'c2': c2, 'w': w},
                          bounds=bounds)

# 初始化记录最佳位置和适应度的列表
best_pos_history = []
fitness_history = []

# 针对每个目标函数执行优化
objective_functions = [objective_function_1, objective_function_2, 
                       objective_function_3, objective_function_4]
fitness_functions = [fitness_function_1, fitness_function_2, 
                     fitness_function_3, fitness_function_4]

# 记录每个目标函数的最佳适应度
best_fitness_history = [[] for _ in range(len(objective_functions))]

for obj_func, fit_func in zip(objective_functions, fitness_functions):
    for _ in range(iters):
        cost, pos = optimizer.optimize(obj_func, iters=1)  # 每次优化一轮
        best_pos_history.append(optimizer.swarm.best_pos.copy())
        fitness = fit_func(cost)
        fitness_history.append(fitness)
        best_fitness_history[objective_functions.index(obj_func)].append(np.min(fitness))

best_pos_history = np.array(best_pos_history)
fitness_history = np.array(fitness_history)

# 获取粒子位置历史
pos_history = np.array(optimizer.pos_history)
pos_history_flat = pos_history.reshape(-1, dimensions)

# PCA降维
pca = PCA(n_components=2)
pos_history_2d = pca.fit_transform(pos_history_flat)

# 将适应度值映射到色条
fitness_values = np.concatenate([np.full(n_particles, fit_func(f)) for f in fitness_history])
fitness_values = np.nan_to_num(fitness_values)  # 将NaN替换为0

# 设置全局作图样式，禁用LaTeX渲染
plt.rcParams['font.family'] = ['SimHei']  # 使用常用字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
plt.rcParams['text.usetex'] = False  # 禁用LaTeX渲染
plt.style.use('seaborn-v0_8-deep')  # 使用Seaborn样式

# 绘制粒子位置的PCA投影
plt.figure(figsize=(10, 6))
plt.scatter(pos_history_2d[:, 0], pos_history_2d[:, 1], c=fitness_values, cmap='viridis')
plt.colorbar(label='适应度')
plt.title("粒子位置的PCA投影", fontsize=16)
plt.xlabel("第一主成分", fontsize=14)
plt.ylabel("第二主成分", fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # 添加网格
plt.minorticks_on()  # 启用次刻度
plt.show()

# 绘制每个目标函数的最佳适应度变化
plt.figure(figsize=(9.35, 6))
for i, best_fitness in enumerate(best_fitness_history):
    plt.plot(best_fitness, label=f'目标函数 {i+1}')
    
# # 设置y轴范围，确保所有曲线都能显示
# plt.ylim(min([min(fitness) for fitness in best_fitness_history]) - 1, 
#           max([max(fitness) for fitness in best_fitness_history]) + 1)

# # 设置y轴范围，确保所有曲线都能显示
# 设置y轴范围，确保所有曲线都能显示
plt.ylim(-30000, 30000)

plt.title("每个目标函数最佳适应度的变化", fontsize=16)
plt.xlabel("迭代次数", fontsize=14)
plt.ylabel("最佳适应度", fontsize=14)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # 添加网格
plt.minorticks_on()  # 启用次刻度
plt.show()