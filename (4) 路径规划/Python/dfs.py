# Dfs 算法
from common import tic, toc, GridMap

from typing import List, Tuple, Union
from functools import lru_cache
from dataclasses import dataclass


# 常量
IMAGE_PATH = 'image1.jpg'  # 图像路径
THRESH = 172              # 二值化阈值
HEIGHT = 350              # 地图高度
WIDTH = 600               # 地图宽度
START = (290, 270)        # 起始坐标
END = (298, 150)          # 结束坐标

# GridMap 初始化
MAP = GridMap(IMAGE_PATH, THRESH, HEIGHT, WIDTH)

@dataclass(eq=False)
class Node:
    """表示网格地图中的一个节点。"""
    x: int
    y: int
    parent: "Node" = None

    def __sub__(self, other: Union["Node", Tuple[int, int]]) -> int:
        """计算曼哈顿距离。"""
        if isinstance(other, Node):
            return abs(self.x - other.x) + abs(self.y - other.y)
        elif isinstance(other, (tuple, list)):
            return abs(self.x - other[0]) + abs(self.y - other[1])
        raise ValueError("other 必须是坐标或 Node")

    def __add__(self, other: Tuple[int, int]) -> "Node":
        """生成一个新节点。"""
        return Node(self.x + other[0], self.y + other[1], self)

    def __eq__(self, other: Union["Node", Tuple[int, int]]) -> bool:
        """基于坐标检查节点是否相等。"""
        if isinstance(other, Node):
            return self.x == other.x and self.y == other.y
        elif isinstance(other, (tuple, list)):
            return self.x == other[0] and self.y == other[1]
        return False

    def __hash__(self) -> int:
        """节点的哈希函数。"""
        return hash((self.x, self.y))

class DFS:
    """深度优先搜索算法实现。"""

    def __init__(self, start_pos: Tuple[int, int] = START, end_pos: Tuple[int, int] = END,
                 map_array=None, move_step: int = 5, move_direction: int = 8):
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
        """重置 DFS 算法。"""
        self.__reset_flag = False
        self.move_step = move_step
        self.move_direction = move_direction
        self.close_set = set()
        self.open_list = []
        self.path_list = []

    def search(self) -> List[Node]:
        """开始搜索。"""
        return self.__call__()

    def _in_map(self, node: Node) -> bool:
        """检查节点是否在网格地图内。"""
        return 0 <= node.x < self.width and 0 <= node.y < self.height

    def _is_collided(self, node: Node) -> bool:
        """检查节点是否在障碍物上。"""
        return self.map_array[node.y, node.x] == 0

    @lru_cache(maxsize=3)
    def _move(self, move_step: int, move_direction: int) -> List[Tuple[int, int]]:
        """获取可能的移动方向。"""
        moves = [
            (0, move_step), (0, -move_step), (-move_step, 0), (move_step, 0),
            (move_step, move_step), (move_step, -move_step), (-move_step, move_step), (-move_step, -move_step)
        ]
        return moves[:move_direction][::-1]

    def _update_open_list(self, curr: Node):
        """更新开放列表。"""
        for add in self._move(self.move_step, self.move_direction):
            next_ = curr + add
            if not self._in_map(next_) or self._is_collided(next_) or next_ in self.close_set or next_ in self.open_list:
                continue
            self.open_list.append(next_)
            if (next_ - self.end) < 20:
                self.move_step = 1

    def __call__(self) -> List[Node]:
        """执行 DFS 查找路径。"""
        assert not self.__reset_flag, "在运行搜索之前请调用 reset"
        print("正在搜索...\n")

        self.open_list.append(self.start)
        
        tic()
        
        while self.open_list:
            curr = self.open_list.pop()
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
    dfs = DFS()
    path = dfs.search()
    MAP.show_path(path)
