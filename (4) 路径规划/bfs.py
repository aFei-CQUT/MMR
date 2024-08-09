# Bfs 算法
from common import GridMap
from common import tic, toc

from typing import Union, List, Tuple
from functools import lru_cache
from collections import deque
from dataclasses import dataclass


# 地图和 BFS 设置常量
IMAGE_PATH = 'image1.jpg'  # 原始图像路径
THRESH = 172              # 二值化阈值
HEIGHT = 350              # 地图高度
WIDTH = 600               # 地图宽度
START = (290, 270)        # 起始坐标
END = (298, 150)          # 结束坐标

MAP = GridMap(IMAGE_PATH, THRESH, HEIGHT, WIDTH)  # GridMap 对象

Number = Union[int, float]

@dataclass(eq=False)
class Node:
    """表示网格中一个点的节点类。"""
    x: int
    y: int
    parent: Union["Node", None] = None

    def __sub__(self, other: Union["Node", Tuple[int, int]]) -> int:
        """计算到另一个节点或坐标的曼哈顿距离。"""
        if isinstance(other, Node):
            return abs(self.x - other.x) + abs(self.y - other.y)
        elif isinstance(other, (tuple, list)):
            return abs(self.x - other[0]) + abs(self.y - other[1])
        raise ValueError("other 必须是 Node 或坐标元组/列表。")
    
    def __add__(self, other: Tuple[int, int]) -> "Node":
        """通过添加移动向量生成一个新节点。"""
        return Node(self.x + other[0], self.y + other[1], self)
        
    def __eq__(self, other: Union["Node", Tuple[int, int]]) -> bool:
        """基于坐标进行相等比较。"""
        if isinstance(other, Node):
            return self.x == other.x and self.y == other.y
        elif isinstance(other, (tuple, list)):
            return self.x == other[0] and self.y == other[1]
        return False
    
    def __hash__(self) -> int:
        """使 Node 可哈希，以便用于集合中。"""
        return hash((self.x, self.y))

class BFS:
    """广度优先搜索 (BFS) 算法实现。"""

    def __init__(
        self,
        start_pos: Tuple[int, int] = START,
        end_pos: Tuple[int, int] = END,
        map_array=None,
        move_step: int = 3,
        move_direction: int = 8,
    ):
        if map_array is None:
            map_array = MAP.map_array

        self.map_array = map_array  # 二值化地图 (0: 障碍物, 255: 自由)
        self.width = self.map_array.shape[1]
        self.height = self.map_array.shape[0]

        self.start = Node(*start_pos)  # 起始位置
        self.end = Node(*end_pos)      # 结束位置

        # 验证起始和结束位置是否有效
        self._validate_positions()

        # 初始化 BFS 算法
        self.reset(move_step, move_direction)

    def _validate_positions(self):
        """验证起始和结束位置是否在地图内且没有在障碍物上。"""
        if not self._in_map(self.start) or not self._in_map(self.end):
            raise ValueError(f"x 坐标范围必须是 0~{self.width-1}，y 坐标范围必须是 0~{self.height-1}")
        if self._is_collided(self.start):
            raise ValueError("起始位置在障碍物上。")
        if self._is_collided(self.end):
            raise ValueError("结束位置在障碍物上。")

    def reset(self, move_step: int = 3, move_direction: int = 8):
        """重置 BFS 算法的状态。"""
        self.__reset_flag = False
        self.move_step = move_step
        self.move_direction = move_direction
        self.close_set = set()
        self.open_deque = deque()
        self.path_list = []

    def search(self) -> List[Node]:
        """从起点到终点搜索路径。"""
        return self.__call__()

    def _in_map(self, node: Node) -> bool:
        """检查节点是否在地图边界内。"""
        return 0 <= node.x < self.width and 0 <= node.y < self.height
    
    def _is_collided(self, node: Node) -> bool:
        """检查节点是否在障碍物上。"""
        return self.map_array[node.y, node.x] == 0
    
    def _move(self) -> List[Tuple[int, int]]:
        """生成基于当前步长和方向的可能移动。"""
        @lru_cache(maxsize=3)
        def _move_cached(move_step: int, move_direction: int) -> List[Tuple[int, int]]:
            moves = [
                (0, move_step),      # 向上
                (0, -move_step),     # 向下
                (-move_step, 0),     # 向左
                (move_step, 0),      # 向右
                (move_step, move_step),    # 上-右
                (move_step, -move_step),   # 下-右
                (-move_step, move_step),   # 上-左
                (-move_step, -move_step),  # 下-左
            ]
            return moves[:move_direction]
        return _move_cached(self.move_step, self.move_direction)

    def _update_open_list(self, curr: Node):
        """将可行的节点添加到开放列表中。"""
        for move in self._move():
            next_node = curr + move

            if not self._in_map(next_node) or self._is_collided(next_node):
                continue
            if next_node in self.close_set or next_node in self.open_deque:
                continue

            self.open_deque.append(next_node)

            if next_node - self.end < 20:
                self.move_step = 1

    def __call__(self) -> List[Node]:
        """执行 BFS 路径查找。"""
        assert not self.__reset_flag, "执行 BFS 之前必须调用 reset。"
        print("正在搜索...\n")

        self.open_deque.append(self.start)

        tic()
        
        while self.open_deque:
            curr = self.open_deque.popleft()
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

if __name__ == '__main__':
    bfs = BFS()
    path = bfs.search()
    MAP.show_path(path)
