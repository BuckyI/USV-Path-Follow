import numpy as np
# import collections
import heapq


class GridWithWeights:
    def __init__(self, img: np.ndarray):
        # 图片矩阵先 y 后 x, 转置之后变成先 x 后 y
        self.maps = img.T  # 存储障碍物信息 True or False
        self.width, self.height = self.maps.shape

    # 这里的设定比较简单, 因为 neighbor 选出来的移动目标只有上下左右, cost 恒为 1
    # 如果有更复杂的移动目标, 可以使用曼哈顿距离或者欧氏距离
    # 作者原版是使用字典存储 self.weights = {}, 适用于更一般的图
    # def cost(self, from_node, to_node):
    #     return self.weights.get(to_node, 1)

    def passable(self, id: tuple) -> bool:
        (x, y) = id
        result = False
        try:
            result = self.maps[x, y]
        except IndexError:
            result = False
        return result

    def neighbors(self, id):
        (x, y) = id
        results = [(x + 1, y), (x - 1, y), (x, y - 1), (x, y + 1)]  # E W N S
        if (x + y) % 2 == 0:
            results.reverse()  # S N W E
        results = filter(self.passable, results)
        return results


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


def heuristic(a, b) -> float:
    "a,b: GridLocation"
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)


def a_star_search(graph: GridWithWeights, start: tuple, goal: tuple):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    shape = (graph.width, graph.height)
    came_from = np.ndarray(shape, dtype=tuple)
    cost_so_far = np.full(shape, np.inf, dtype=np.float16)  # 未探索过的区域为无穷
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()  # Location

        if current == goal:
            break

        for next in graph.neighbors(current):
            # new_cost = cost_so_far[current] + graph.cost(current, next)
            new_cost = cost_so_far[current] + 1  # 由于这里只会上下左右移动, 简化
            if new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(next, goal)
                frontier.put(next, priority)
                came_from[next] = current

    return came_from, cost_so_far


def reconstruct_path(came_from: np.ndarray, start: tuple, goal: tuple):
    current = goal
    path = []

    while current != start:
        path.append(current)
        current = came_from[current]
        if current is None:
            raise ("看起来没有找到路径")

    path.append(start)
    path.reverse()
    return path
