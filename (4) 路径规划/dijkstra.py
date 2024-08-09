# 迪杰斯特拉(Dijkstra)算法
# A*: F = G + H
# Dijkstra: F = G
# https://zhuanlan.zhihu.com/p/346666812

from common import GridMap, SetQueue, ListQueue, PriorityQueuePro, Node
from common import tic, toc

from typing import List, Tuple
from functools import lru_cache


# 队列类型选项
Queue_Type = 2
"""
0 -> SetQueue
1 -> ListQueue
2 -> PriorityQueuePro
"""

# 常量
IMAGE_PATH = 'image1.jpg'  # 图像路径
THRESH = 172              # 二值化阈值
HEIGHT = 350              # 地图高度
WIDTH = 600               # 地图宽度
START = (290, 270)        # 起始坐标
END = (298, 150)          # 结束坐标

# GridMap 初始化
MAP = GridMap(IMAGE_PATH, THRESH, HEIGHT, WIDTH)

# 根据选择的队列类型初始化 OpenList
if Queue_Type == 0:
    NodeQueue = SetQueue
elif Queue_Type == 1:
    NodeQueue = ListQueue
else:
    NodeQueue = PriorityQueuePro

class Dijkstra:
    """Dijkstra 算法用于路径寻找。"""

    def __init__(
        self,
        start_pos: Tuple[int, int] = START,
        end_pos: Tuple[int, int] = END,
        map_array=None,
        move_step: int = 3,
        move_direction: int = 8
    ):
        """初始化算法参数。

        参数
        ----------
        start_pos : Tuple[int, int]
            起始坐标。
        end_pos : Tuple[int, int]
            结束坐标。
        map_array : ndarray
            二值化地图，其中 0 为障碍物，255 为自由空间。
        move_step : int
            移动步长。
        move_direction : int
            移动方向数量（4 或 8）。
        """
        self.map_array = map_array if map_array is not None else MAP.map_array
        self.width = self.map_array.shape[1]
        self.height = self.map_array.shape[0]
        self.start = Node(*start_pos)
        self.end = Node(*end_pos)

        if not self._in_map(self.start) or not self._in_map(self.end):
            raise ValueError(f"x 范围: 0~{self.width-1}, y 范围: 0~{self.height-1}")
        if self._is_collided(self.start) or self._is_collided(self.end):
            raise ValueError("起始或结束位置在障碍物上")

        self.reset(move_step, move_direction)

    def reset(self, move_step: int = 3, move_direction: int = 8):
        """重置算法状态。"""
        self.__reset_flag = False
        self.move_step = move_step
        self.move_direction = move_direction
        self.close_set = set()
        self.open_queue = NodeQueue()
        self.path_list = []

    def search(self) -> List[Node]:
        """执行搜索以找到路径。"""
        return self.__call__()

    def _in_map(self, node: Node) -> bool:
        """检查节点是否在地图边界内。"""
        return 0 <= node.x < self.width and 0 <= node.y < self.height

    def _is_collided(self, node: Node) -> bool:
        """检查节点是否在障碍物上。"""
        return self.map_array[node.y, node.x] < 1

    @lru_cache(maxsize=3)
    def _move(self, move_step: int, move_direction: int) -> List[Tuple[int, int]]:
        """获取基于步长和方向的移动偏移量。"""
        moves = [
            (0, move_step), (0, -move_step), (-move_step, 0), (move_step, 0),
            (move_step, move_step), (move_step, -move_step), (-move_step, move_step), (-move_step, -move_step)
        ]
        return moves[:move_direction]

    def _update_open_list(self, curr: Node):
        """更新开放列表。"""
        for add in self._move(self.move_step, self.move_direction):
            next_ = curr + add
            if not self._in_map(next_) or self._is_collided(next_) or next_ in self.close_set:
                continue
            self.open_queue.put(next_)
            if next_ - self.end < 20:
                self.move_step = 1

    def __call__(self) -> List[Node]:
        """执行 Dijkstra 路径寻找算法。"""
        assert not self.__reset_flag, "在运行搜索之前请调用 reset。"
        print("正在搜索...\n")

        self.open_queue.put(self.start)
        tic()
        while not self.open_queue.empty():
            curr = self.open_queue.get()
            self._update_open_list(curr)
            self.close_set.add(curr)
            if curr == self.end:
                break
        print("搜索完成\n")
        toc()

        while curr.parent is not None:
            self.path_list.append(curr)
            curr = curr.parent
        self.path_list.reverse()

        self.__reset_flag = True
        return self.path_list

# 调试
if __name__ == '__main__':
    dijkstra = Dijkstra()
    path = dijkstra.search()
    MAP.show_path(path)
