import numpy as np
import matplotlib.pyplot as plt
from pyswarms.single.global_best import GlobalBestPSO
from sklearn.decomposition import PCA
import logging

# 配置日志
logging.getLogger('pyswarms').setLevel(logging.WARNING)

# 定义10维的Rosenbrock函数
def rosenbrock_10d(x):
    return np.sum(100.0 * (x[:, 1:] - x[:, :-1]**2.0)**2.0 + (1 - x[:, :-1])**2.0, axis=1)

# 设置优化参数
dimensions = 10
c1, c2, w = 0.5, 0.3, 0.9
n_particles = 50
iters = 1000

# 设置搜索空间边界
bounds = (np.array([-5]*dimensions), np.array([5]*dimensions))

# 初始化优化器
optimizer = GlobalBestPSO(n_particles=n_particles, dimensions=dimensions, 
                          options={'c1': c1, 'c2': c2, 'w': w},
                          bounds=bounds)

# 初始化记录最佳位置的列表
best_pos_history = []

# 执行优化并记录最佳位置
for _ in range(iters):
    cost, pos = optimizer.optimize(rosenbrock_10d, iters=1)  # 每次优化一轮
    best_pos_history.append(optimizer.swarm.best_pos.copy())

best_pos_history = np.array(best_pos_history)

# 获取粒子位置历史
pos_history = np.array(optimizer.pos_history)
pos_history_flat = pos_history.reshape(-1, dimensions)

# PCA降维
pca = PCA(n_components=2)
pos_history_2d = pca.fit_transform(pos_history_flat)

# 设置全局作图样式，禁用LaTeX渲染
plt.rcParams['font.family'] = ['SimHei']  # 使用常用字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
plt.rcParams['text.usetex'] = False  # 禁用LaTeX渲染
plt.style.use('seaborn-v0_8-deep')  # 使用Seaborn样式

# 绘制粒子位置的PCA投影
plt.figure(figsize=(10, 5))
plt.scatter(pos_history_2d[:, 0], pos_history_2d[:, 1], c=np.arange(pos_history_2d.shape[0]), cmap='viridis')
plt.colorbar(label='迭代次数')
plt.title("粒子位置的PCA投影", fontsize=16)
plt.xlabel("第一主成分", fontsize=14)
plt.ylabel("第二主成分", fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # 添加网格
plt.minorticks_on()  # 启用次刻度
plt.show()

# 绘制每个维度中最佳粒子位置的变化
plt.figure(figsize=(9.58, 5))
for i in range(dimensions):
    plt.plot(best_pos_history[:, i], label=f'维度 {i+1}')
plt.title("每个维度中最佳粒子位置的变化", fontsize=16)
plt.xlabel("迭代次数", fontsize=14)
plt.ylabel("位置", fontsize=14)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # 添加网格
plt.minorticks_on()  # 启用次刻度
plt.show()