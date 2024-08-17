import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 创建图的邻接矩阵
A = np.array([[0, 9, 2, 4, 7],
              [9, 0, 3, 4, 0],
              [2, 3, 0, 8, 4],
              [4, 4, 8, 0, 6],
              [7, 0, 4, 6, 0]])

# 创建原始图
G = nx.Graph(A)

# 使用 Kruskal 算法计算最小生成树
T = nx.minimum_spanning_tree(G, algorithm='kruskal')

# 设置布局
pos = nx.spring_layout(G, iterations=20)

# 设置参数
plt.style.use('fivethirtyeight')

# 绘制图
plt.figure(figsize=(10, 6))

# 绘制原始图
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
nx.draw_networkx_edges(G, pos, edge_color='lightgray')
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

# 绘制最小生成树
nx.draw_networkx_edges(T, pos, edge_color='coral', width=2)
nx.draw_networkx_edge_labels(T, pos, edge_labels=nx.get_edge_attributes(T, 'weight'))

# 显示图
plt.title('Original Graph and Minimum Spanning Tree')
plt.show()
