---
aliases: 
tags: 
id: "code_file"
date created: 2022-02-09 20:28:12
date modified: 2022-05-10 21:58:23
---

# A STAR

## 1 原理与实现方法

[A *寻路算法详解 #A 星 #启发式搜索 bilibili](https://www.bilibili.com/video/BV1bv411y79P?from=search&seid=3064739554010128000&spm_id_from=333.337.0.0)
[Implementation of A*](https://www.redblobgames.com/pathfinding/a-star/implementation.html#troubleshooting-ugly-path)
Python 版本的实现看 `1. Python Implementation` 全部部分.
作者讲的非常详细，从简单到复杂，广度优先算法-->Djakarta 算法-->A \*算法. 其中有一些思想和改进建议.

从一个一般性原理来说，写代码需要准备以下三个部分，然后就使用作者提供的`a_star_search`实现就可以了.
> Knowledge about grids is in the graph class (GridWithWeights), the locations, and in the heuristic function. Replace those three and you can use the A* algorithm code with any other graph structure.

- the graph class: 图的信息，结点如何相连之类的
- the locations: 如何定义一个结点
- the heuristic function

实际上，需要自己根据教程，**结合具体情况实现代码**，而非拿现成的.

上面教程的翻译版本，可以看看：
[【翻译】由浅入深 A* 算法介绍 - 知乎](https://zhuanlan.zhihu.com/p/113008274)
[【翻译】A* 算法实现 - 知乎](https://zhuanlan.zhihu.com/p/113390876)

### 1.1 参考代码

`Implementation of A*` 里面有 `implementation.py` 的代码 [链接](https://www.redblobgames.com/pathfinding/a-star/implementation.py) ，不过由于有一定年代了，现在应该不能直接用. 之前 `typing` 包含的东西被移动到了其他地方或者干脆变成了全局通用的. （例如之前的 `typing.Dict` 变成了 `dict`)

```Python
from __future__ import annotations
##some of these types are deprecated: https://www.python.org/dev/peps/pep-0585/
from typing import Protocol, Dict, List, Iterator, Tuple, TypeVar, Optional
```

`【翻译】A* 算法实现 - 知乎` 里面有一份别人学习实现的代码（和原版的略有不同，改正了过时的内容，但是有一点删减）
[Notes/GameDevelopment/GameDevelop/Codes/AStar 算法实现 at master · yangruihan/Notes](https://github.com/yangruihan/Notes/tree/master/GameDevelopment/GameDevelop/Codes/AStar%E7%AE%97%E6%B3%95%E5%AE%9E%E7%8E%B0)

✨✨✨ **代码实现举例** [Astar.7z](assets/Astar/Astar.7z)

### 1.2 其他参考资料

[Amit’s Game Programming Information](http://www-cs-students.stanford.edu/~amitp/gameprog.html#Paths) 找了好久才才发现有这个页面… 有很多值得学习的东西.
作者的 GitHub
[redblobgames (Amit Patel)](https://github.com/redblobgames)
[amitp (Amit Patel)](https://github.com/amitp)

---
[路径规划之 A* 算法](https://paul.pub/a-star-algorithm/) 差不多是另一个实现，没仔细看.

## 2 具体实现

### 2.1 原版

下面这段代码直接来源于作者的说明以及搜集来的代码（都在上一节）

- the graph class: `GridWithWeights`, 一个栅格化的图，只存储长宽，还有障碍物的位置
- the locations: 其实就是元组 `(x, y)`
- the heuristic function: 当前位置与目标点的曼哈顿距离

> 注意：
> 据作者所说，很多实现用到了 Node 类，但是他倾向于使用字典存储（上一个节点，当前代价）
> 当然如果是二维栅格图，可以使用矩阵存储（后面有一版）

```python
import numpy as np
import heapq

class GridWithWeights:
    def __init__(self, width, height):
        self.weights = {}
        self.width = width
        self.height = height
        self.walls = []

    def cost(self, from_node, to_node):
        return self.weights.get(to_node, 1)

    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, id):
        return id not in self.walls

    def neighbors(self, id):
        (x, y) = id
        results = [(x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1)]
        if (x + y) % 2 == 0:
            results.reverse()
        results = filter(self.in_bounds, results)
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

def a_star_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}  # : Dict[Location, Optional[Location]]
    cost_so_far = {}  # : Dict[Location, float]
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()  # Location

        if current == goal:
            break

        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(next, goal)
                frontier.put(next, priority)
                came_from[next] = current

    return came_from, cost_so_far
```

#### 2.1.1 Tips

有一步需要用 passable 函数过滤掉那些不应该成为下一步的位置（超出边界或者属于障碍物）.
我把 `in_bounds` 和 `passable` 合并了，但是要注意 `return` 后面表达式**先后位置**. `and` 有短路现象，先判断第一个，如果为 False, 就不会计算第二个表达式了.

```Python
def passable(self, id: tuple) -> bool:
    (x, y) = id
    return 0 <= x < self.width and 0 <= y < self.height and self.maps[x, y]
```

> 另外 jupyter notebook 里面实际操作起来有点不一样… 还是会报错不知道为啥

进一步其实这样写更简单, 节省了两个变量

```Python
def passable(self, id: tuple) -> bool:
    (x, y) = id
    try:
        return self.maps[x, y]
    except IndexError:
        return False
```

### 2.2 对图片矩阵使用 A\*算法

一个二值化的图像 (0 表示障碍物，1 表示可以通行）, 指定起止像素坐标寻路.
和原始实现基本相同，只不过把一些地方的数据类型改成了 numpy 的数组，而非字典/列表.

比如`came_from`存储某个结点的上一个结点，之前字典是类似 `{(x,y): (i,j)}`, 要改成 numpy 的话，就是让一个二维数组存储的元素类型为元组. `np.ndarray((2, 3), dtype=tuple)`
> 但是这里实际上获得的矩阵中，`dtype=object`, 任何对象都可以存.

更进一步，`np.ndarray((2, 3), dtype=np.dtype("2i"))` 的含义是，存储的元素是 `(2,)` 形状的 array, 数据类型是 `i` 整数.
其实获得的是 `shape=(2,3,2)` 的 array.
> 我感觉第二种效率更高，但是由于算法中要求没有访问过的结点其 `came_from` 对应是 None, 就选了第一种方法

`cost_so_far` 要求创建带默认值的矩阵/字典（作者有具体解释）
> `numpy.full[shape, fill_value, dtype=None, order='C', *, like=None](source)`
> :   Return a new array of given shape and type, filled with fill_value.

### 2.3 Troubleshooting

如果没有找到（不存在）路径，队列为空函数返回. 可以命令行打印一个 success 表示函数是从 break 退出的.

## 3 代码实现（旧）

不是特别熟悉，有网上搜集到的别人的程序 [medium.com](https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2)
（似乎实现的不是很高明，但也可以用，有时候会花费很久时间还寻路失败）, 下面有一段使用代码
> 我好像把源代码稍微改了改，也可能没改，总之这里使用 ndarray 作为交互数据结构，输出的路径本来是列表，转化为 ndarray
> 这个也是找到的一个实现，没有试 [zhijs/8puzzle: 基于 Python 八数码问题算法（深广度算法，A 星算法）](https://github.com/zhijs/8puzzle)

```python
from pyastar import a_star
## start,end=np.array([440, 50]),np.array([440, 130])
start,end=targets[0,:],targets[3,:]
print(start,end)
print(maze[start[1],start[0]])
print(maze[end[1],end[0]])

path = a_star(maze, start,end)
path= np.array(path)
```

## 4 关于路径长度的讨论

这里预设存储路径的是一个 nx2 矩阵，从起点到终点的坐标.

单向路径
上一个点减去下一个点，所以需要错一位取矩阵相减.
`np.linalg.norm` 求模长（大概）, 之后`np.sum`求和

```python
diff = np.array(path[:-1]) - np.array(path[1:])  # 先获得相邻的差
np.sum(np.linalg.norm(diff, axis=1))
```

> 特别注意的是，对于 A star 算法，如果只有上下左右四个方向，每走一步 cost 都是 1, 那么只需要 `len(path)-1`

环形路径

```python
diff=np.linalg.norm(overall_path-np.roll(overall_path, 1, axis=0), axis=1) # 先获得相邻的差
path_lenghth=np.sum(diff) # 闭合路径
print(path_lenghth,"km")
```
