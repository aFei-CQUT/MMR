# %% 导入库
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# %% 数据准备
# 源数据
df = pd.DataFrame({
    'year': [1790, 1800, 1810, 1820, 1830, 1840, 1850, 1860, 1870],
    'population': [3.9, 5.3, 7.2, 9.6, 12.9, 17.1, 23.2, 31.4, 38.6],
})

x0 = df['population'][0]
t0 = df['year'][0]

# %% 定义改进的 Logistic 模型
def improved_logistic_model(t, r0, r1, xm):
    r = r0 + r1 * (t - t0)  # 动态增长率
    return xm / (1 + (xm/x0 - 1) * np.exp(-r * (t - t0)))

# %% 拟合参数
initial_guess = [0.1, 0.01, 50]  # 初始猜测
bounds = (0, [1, 1, 100])  # 参数边界
popt, pcov = curve_fit(improved_logistic_model, df['year'], df['population'], p0=initial_guess, bounds=bounds)
r0, r1, xm = popt
print('r0 =', r0)
print('r1 =', r1)
print('xm =', xm)

# %% 预测 1900 年的人口
predicted_population_1900 = improved_logistic_model(1900, r0, r1, xm)
print('Population in 1900 =', predicted_population_1900)

# %% 可视化预测结果
year = np.linspace(1790, 2000, 211)
population = improved_logistic_model(year, r0, r1, xm)

# 设置全局样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['text.usetex'] = False  # 禁用 LaTeX 渲染

plt.figure(figsize=(10, 6))  # 设置图像大小
plt.scatter(df['year'], df['population'], label='实际数据', color='blue', s=50)  # 实际数据点
plt.plot(year, population, label='改进的 Logistic 模型预测', color='coral', linewidth=2)  # 模型预测曲线
plt.title('改进的 Logistic 模型预测人口增长', fontsize=16)
plt.xlabel('年份', fontsize=14)
plt.ylabel('人口（百万）', fontsize=14)
plt.legend(fontsize=12)
plt.grid()
plt.show()
