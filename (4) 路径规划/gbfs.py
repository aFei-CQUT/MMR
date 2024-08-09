# 贪婪最佳优先搜索(Greedy Best First Search, GBFS)算法 
# A*: F = G + H
# GBFS: F = H
# https://zhuanlan.zhihu.com/p/346666812
from common import GridMap, SetQueue, ListQueue, PriorityQueuePro, Node
from common import tic, toc

from functools import lru_cache


Queue_Type = 0
"""
# OpenList 队列类型
## 0 -> SetQueue
## 1 -> ListQueue
## 2 -> PriorityQueuePro
# - SetQueue: 更新父节点和成本的效率高，可能找到更好的路径。
# - ListQueue: 最慢，但也允许更新。
# - PriorityQueuePro: 速度最快但不能更新信息，可能导致路径质量较差。
"""

# 地图设置
IMAGE_PATH = 'image1.jpg'  # 原始图像路径
THRESH = 172              # 二值化阈值（值大于 THRESH 变为 255，否则为 0）
HEIGHT = 350              # 地图高度
WIDTH = 600               # 地图宽度

MAP = GridMap(IMAGE_PATH, THRESH, HEIGHT, WIDTH)  # GridMap 对象

# 起点和终点
START = (290, 270)  # 起始坐标（y 轴向下为正方向）
END = (298, 150)    # 结束坐标（y 轴向下为正方向）

""" ---------------------------- Greedy Best First Search 算法 ---------------------------- """
# F = H

# 根据 Queue_Type 设置 OpenList 队列
if Queue_Type == 0:
    NodeQueue = SetQueue
elif Queue_Type == 1:
    NodeQueue = ListQueue
else:
    NodeQueue = PriorityQueuePro

class GBFS:
    """Greedy Best First Search 算法"""

    def __init__(self, start_pos=START, end_pos=END, map_array=MAP.map_array, move_step=3, move_direction=8):
        """
        初始化 GBFS 算法。

        参数
        ----------
        start_pos : tuple/list
            起始坐标。
        end_pos : tuple/list
            结束坐标。
        map_array : ndarray
            二值化地图，0 表示障碍物，255 表示自由空间，大小为 H*W。
        move_step : int
            移动步长，默认值为 3。
        move_direction : int (8 或 4)
            移动方向，默认值为 8 个方向。
        """
        self.map_array = map_array  # H * W 的网格地图
        self.width = self.map_array.shape[1]
        self.height = self.map_array.shape[0]
        self.start = Node(*start_pos)  # 起点
        self.end = Node(*end_pos)      # 终点

        # 错误检查
        if not self._in_map(self.start) or not self._in_map(self.end):
            raise ValueError(f"x 坐标范围必须在 0~{self.width-1}，y 坐标范围在 0~{self.height-1}")
        if self._is_collided(self.start):
            raise ValueError("起始位置在障碍物上")
        if self._is_collided(self.end):
            raise ValueError("结束位置在障碍物上")

        # 初始化算法
        self.reset(move_step, move_direction)

    def reset(self, move_step=3, move_direction=8):
        """重置算法状态。"""
        self.__reset_flag = False
        self.move_step = move_step                # 移动步长（搜索过程中会减少）
        self.move_direction = move_direction      # 移动方向数量（4 或 8）
        self.close_set = set()                    # 已访问节点的集合
        self.open_queue = NodeQueue()             # 开放列表的优先队列
        self.path_list = []                       # 存储最终路径的列表

    def search(self):
        """执行路径搜索。"""
        return self.__call__()

    def _in_map(self, node: Node):
        """检查节点是否在地图边界内。"""
        return (0 <= node.x < self.width) and (0 <= node.y < self.height)

    def _is_collided(self, node: Node):
        """检查节点是否与障碍物碰撞。"""
        return self.map_array[node.y, node.x] == 0

    def _move(self):
        """获取基于步长和方向的移动偏移量。"""
        @lru_cache(maxsize=3)  # 使用缓存避免重复计算
        def _move(move_step: int, move_direction: int):
            moves = (
                [0, move_step],        # 向上
                [0, -move_step],       # 向下
                [-move_step, 0],       # 向左
                [move_step, 0],        # 向右
                [move_step, move_step],        # 向右上
                [move_step, -move_step],       # 向右下
                [-move_step, move_step],       # 向左上
                [-move_step, -move_step],      # 向左下
            )
            return moves[:move_direction]
        return _move(self.move_step, self.move_direction)

    def _update_open_list(self, curr: Node):
        """更新开放列表中的有效相邻节点。"""
        for add in self._move():
            next_ = curr + add

            if not self._in_map(next_) or self._is_collided(next_) or next_ in self.close_set:
                continue

            # 计算新节点的代价
            H = next_ - self.end  # 估算的剩余距离
            next_.cost = H  # G = 0

            # 将节点添加到开放列表中或更新节点
            self.open_queue.put(next_)

            # 当接近目标时减少步长
            if H < 20:
                self.move_step = 1

    def __call__(self):
        """执行 GBFS 路径搜索。"""
        assert not self.__reset_flag, "在开始搜索之前请调用 reset"
        print("正在搜索...\n")

        # 初始化开放列表
        self.open_queue.put(self.start)

        # 搜索循环
        tic()
        while not self.open_queue.empty():
            curr = self.open_queue.get()  # 获取 H 值最小的节点
            self._update_open_list(curr)
            self.close_set.add(curr)  # 添加到闭合列表

            if curr == self.end:
                break
        print("路径搜索完成\n")
        toc()

        # 重建路径
        while curr.parent is not None:
            self.path_list.append(curr)
            curr = curr.parent
        self.path_list.reverse()

        # 重置标志
        self.__reset_flag = True

        return self.path_list

# 调试
if __name__ == '__main__':
    gbfs = GBFS()
    path = gbfs.search()
    MAP.show_path(path)
