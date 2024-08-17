import networkx as nx
import matplotlib.pyplot as plt

# 创建有向图
G = nx.DiGraph()

# 定义边列表
edge_list = [
    ('v1', 'v2', 3),
    ('v1', 'v3', 2),
    ('v2', 'v4', 4),
    ('v2', 'v5', 6),
    ('v2', 'v6', 2),
    ('v3', 'v2', 8),
    ('v3', 'v4', 1),
    ('v3', 'v6', 3),
    ('v4', 'v5', 2),
    ('v6', 'v5', 5),
    ('v5', 'v7', 7),
    ('v6', 'v7', 3),
]

# 添加边到图中
for edge in edge_list:
    G.add_edge(edge[0], edge[1], capacity=edge[2])

# 计算最大流
value, flow_dic = nx.maximum_flow(G, 'v1', 'v7')

# 打印最大流值和流量字典
print("Maximum flow value:", value)
print("Flow dictionary:", flow_dic)

# 准备边标签
dic = {}
for s in flow_dic:
    for t in flow_dic[s]:
        dic[(s, t)] = flow_dic[s][t]

# 设置布局
pos = nx.spring_layout(G)

# # 设置参数
# plt.style.use('ggplot')

# 绘制原始图
plt.figure(figsize=(10, 6))
nx.draw_networkx(G, pos=pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
plt.title('Original Directed Graph')
plt.show()

# 绘制最大流图
plt.figure(figsize=(10, 6))

# 绘制原始图的边（浅灰色）
nx.draw_networkx_edges(G, pos, edge_color='lightgray', alpha=0.5)

# 绘制所有节点
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)

# 绘制节点标签
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

# 绘制最大流的边（使用不同的颜色）
for s in flow_dic:
    for t in flow_dic[s]:
        if flow_dic[s][t] > 0:  # 只绘制流量大于0的边
            nx.draw_networkx_edges(G, pos, edgelist=[(s, t)], edge_color='coral', width=2)

# 绘制流量标签
nx.draw_networkx_edge_labels(G, pos, edge_labels=dic)

# 显示图形
plt.title('Directed Graph with Maximum Flow')
plt.show()
