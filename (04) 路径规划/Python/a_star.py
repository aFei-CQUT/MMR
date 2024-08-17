# A*算法 
# http://www.360doc.com/content/21/0811/13/40892371_990562118.shtml

from common import GridMap, SetQueue, ListQueue, PriorityQueuePro, Node
from common import tic, toc

from functools import lru_cache


# 队列类型常量
SET_QUEUE = 0
LIST_QUEUE = 1
PRIORITY_QUEUE_PRO = 2

# 配置
QUEUE_TYPE = SET_QUEUE
IMAGE_PATH = 'image1.jpg'  # 原始图像路径
THRESH = 172              # 图像二值化的阈值
HEIGHT = 350              # 地图高度
WIDTH = 600               # 地图宽度

# 初始化网格地图
MAP = GridMap(IMAGE_PATH, THRESH, HEIGHT, WIDTH)

# 起始和结束位置
START = (290, 270)  # 起始位置（y轴正方向向下）
END = (298, 150)    # 结束位置（y轴正方向向下）

# 根据队列类型选择合适的队列
if QUEUE_TYPE == SET_QUEUE:
    NodeQueue = SetQueue
elif QUEUE_TYPE == LIST_QUEUE:
    NodeQueue = ListQueue
else:
    NodeQueue = PriorityQueuePro

class AStar:
    """A* 算法实现"""

    def __init__(self, start_pos=START, end_pos=END, map_array=MAP.map_array, move_step=3, move_direction=8):
        """
        使用参数初始化 A* 算法。

        参数
        ----------
        start_pos : tuple
            起始位置坐标。
        end_pos : tuple
            结束位置坐标。
        map_array : ndarray
            二值化地图，0 代表障碍物，255 代表自由空间。
        move_step : int
            移动步长。默认为 3。
        move_direction : int
            移动方向数。默认为 8。
        """
        self.map_array = map_array
        self.width = self.map_array.shape[1]
        self.height = self.map_array.shape[0]
        self.start = Node(*start_pos)
        self.end = Node(*end_pos)

        # 验证起始和结束位置
        self._validate_position(self.start, "Start")
        self._validate_position(self.end, "End")

        # 初始化算法
        self.reset(move_step, move_direction)

    def _validate_position(self, node: Node, name: str):
        """验证位置是否在地图内并且没有与障碍物碰撞。"""
        if not self._in_map(node):
            raise ValueError(f"{name} x 坐标范围是 0~{self.width - 1}，y 坐标范围是 0~{self.height - 1}")
        if self._is_collided(node):
            raise ValueError(f"{name} x 或 y 坐标在障碍物上")

    def reset(self, move_step=3, move_direction=8):
        """重置算法以进行新的搜索。"""
        self.__reset_flag = False
        self.move_step = move_step
        self.move_direction = move_direction
        self.closed_set = set()
        self.open_queue = NodeQueue()
        self.path_list = []

    def search(self):
        """开始路径搜索。"""
        return self.__call__()

    def _in_map(self, node: Node):
        """检查节点是否在地图边界内。"""
        return 0 <= node.x < self.width and 0 <= node.y < self.height

    def _is_collided(self, node: Node):
        """检查节点是否与障碍物碰撞。"""
        return self.map_array[node.y, node.x] == 0

    @lru_cache(maxsize=3)
    def _get_movements(self):
        """生成基于步长和方向的可能移动。"""
        movements = [
            (0, self.move_step),   # 向上
            (0, -self.move_step),  # 向下
            (-self.move_step, 0),  # 向左
            (self.move_step, 0),   # 向右
            (self.move_step, self.move_step),   # 上-右
            (self.move_step, -self.move_step),  # 下-右
            (-self.move_step, self.move_step),  # 上-左
            (-self.move_step, -self.move_step), # 下-左
        ]
        return movements[:self.move_direction]

    def _update_open_list(self, current_node: Node):
        """将有效的邻居节点添加到开放列表中。"""
        for move in self._get_movements():
            next_node = current_node + move

            if not self._in_map(next_node) or self._is_collided(next_node) or next_node in self.closed_set:
                continue

            next_node.cost += next_node - self.end  # F = G + H

            self.open_queue.put(next_node)

            if next_node - self.end < 20:
                self.move_step = 1

    def __call__(self):
        """执行 A* 路径搜索。"""
        if self.__reset_flag:
            raise RuntimeError("请在调用搜索之前重置算法")

        print("正在搜索...\n")

        self.open_queue.put(self.start)

        tic()
        
        while not self.open_queue.empty():
            current_node = self.open_queue.get()
            current_node.cost -= current_node - self.end

            self._update_open_list(current_node)
            self.closed_set.add(current_node)

            if current_node == self.end:
                break
            
        print("搜索完成\n")
        
        toc()
        
        self._construct_path(current_node)
        self.__reset_flag = True

        return self.path_list

    def _construct_path(self, current_node: Node):
        """通过从结束节点回溯构造路径。"""
        while current_node.parent:
            self.path_list.append(current_node)
            current_node = current_node.parent
        self.path_list.reverse()


# 调试
if __name__ == '__main__':
    astar = AStar()
    path = astar.search()
    MAP.show_path(path)
