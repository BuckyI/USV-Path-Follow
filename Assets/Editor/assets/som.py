import numpy as np
import random
"""提供一个 Network, 输入地图, 目标点, 起点, 终点, 获得路径"""

# math operation function


def ver_vec(direction, vector):
    """
    向量垂直分解
    direction: [ndarray] 方向向量
    vector: [ndarray] 要分解的向量
    return: vector 垂直于 direction 的分向量.
    两个array相同行,对应行求垂直向量.
    """
    x0, y0 = direction[:, 0], direction[:, 1]
    x1, y1 = vector[:, 0], vector[:, 1]
    x = y0**2 * x1 - x0 * y0 * y1
    y = x0**2 * y1 - x0 * y0 * x1
    return np.array([x, y]).T / (x0**2 + y0**2)[:, np.newaxis]


def unit_vector(vector):
    "vector.shape==(n,2) 返回单位向量"
    v = vector.copy()
    a = np.linalg.norm(v, axis=1, keepdims=True)
    indices = a[:, 0] != 0  # 找到不是零向量的
    v[indices] = v[indices] / a[indices]
    return v


class Network():
    def __init__(self, targets, maze, start, end):
        self.start, self.end = start, end
        self.targets = targets
        self.maze = maze  # True 是可通行, False 是不可通行
        self.node_number = targets.shape[0] * 15
        # 生成神经网络
        center = targets.sum(axis=0) / len(targets)
        self.network = np.random.rand(self.node_number, 2) * center
        # 用在邻域函数里面的邻域半径
        self.radius = self.node_number
        self.learning_rate = 0.8  # 初始学习率设为0.8

    def training(self):
        for _ in range(100000):
            # 设置起点终点
            self.network[0] = self.start
            self.network[-1] = self.end

            self.traditional_som()
            self.window_som()

            self.learning_rate *= 0.99997
            self.radius *= 0.9997

            if self.radius < 1:
                break
            elif self.learning_rate < 0.001:
                break

    def traditional_som(self):
        target = random.choice(self.targets)
        winner_idx = self.select_closest(self.network, target)
        gaussian = self.get_neighborhood(winner_idx, self.radius // 10)
        target_delta = gaussian[:, np.newaxis] * (target - self.network)
        self.network += self.learning_rate * target_delta
        return self.network

    def window_som(self):
        # 去头去尾, 只有中间的点才避障
        m = self.network[1:-1]
        s = self.network[:-2]
        d = self.network[2:]
        sd = d - s  # 衡量区域宽度,对区域进行分割的依据
        sm = m - s
        # step length
        step = (np.linalg.norm(sd, axis=1, keepdims=True)).clip(
            1, 10) * self.learning_rate  # 一步的步长
        # sm 垂直 sd 分解, 如果 sd 是零向量, 则认为 sm 近似为垂直方向
        vm = ver_vec(sd, sm)
        vm[s == d] = sm[s == d]
        # step direction
        unit_head_dir = unit_vector(-vm)

        base_net = s + 0.5 * sd + self.learning_rate * vm

        result = self.get_away(
            network=base_net,
            step=step,
            head_dir=unit_head_dir,
            k=0,
            max_k=10
            # max_k=10 / self.learning_rate,  # 后面单个步长短但是最大总步长变大
            # max_k=np.inf,  # 为了躲避障碍物,会一直走下去
        )

        # 胜者不改变
        winner_indices = np.apply_along_axis(
            func1d=lambda t: self.select_closest(self.network, t),
            axis=1,
            arr=self.targets,
        )
        winners = self.network[winner_indices].copy()

        self.network[1:-1] = result  # 保存窗口更新结果
        self.network[winner_indices] = winners  # 恢复胜者原来位置

    def select_closest(self, candidates, origin):
        """
        candidates: [numpy.ndarray] size*2 it is actually all the neurons\n
        origin: [numpy.ndarray] 1*2 it is the chosen city\n
        return: [int] at the given city, the nearest neuron's index\n

        检索array中最小值的位置，并返回其下标值，即找到最相似neuron\n
        Return the index of the closest candidate to a given point.\n
        """
        return self.euclidean_distance(candidates, origin).argmin()

    def euclidean_distance(self, a, b):
        """
        a, b = candidates, origin\n
        求a-b每一行的算数平方和开根号，也就是行向量之间的欧几里德距离了。\n
        Return the array of distances of two numpy arrays of points.\n
        """
        return np.linalg.norm(a - b, axis=1)

    def route_distance(self):
        """
        calculate current route distance
        Return the cost of traversing a route of cities in a certain order.
        """
        distances = self.euclidean_distance(self.network[:-1],
                                            self.network[1:])
        return np.sum(distances)

    def get_neighborhood(self, center, radix):
        """
        Get the range gaussian of given radix around a center index.
        """
        # Impose an upper bound on the radix to prevent NaN and blocks
        if radix < 1:
            radix = 1
        # 非环形拓扑结构下的 distance
        delta = np.absolute(center - np.arange(self.node_number))
        # distances = np.minimum(deltas, self.node_number - deltas)

        # Compute Gaussian distribution around the given center
        return np.exp(-delta**2 / (2 * radix**2))

    def troubled_nodes(self, network):
        def troubled_point(x, y, maze):
            if x >= maze.shape[1] or y >= maze.shape[0]:
                return True
            else:
                return not maze[y, x]

        ind_node = np.apply_along_axis(
            func1d=lambda p: troubled_point(int(p[0]), int(p[1]), self.maze
                                            ),  # 0 为障碍物
            axis=1,
            arr=network)
        return ind_node

    def get_away(self, network, step, head_dir, k=0, max_k=5):
        "network 沿着 dir 方向脱离障碍物, 最大步长为 max_k*step"

        # step 1 update and find the bad nodes
        if k == 0:
            # k==0 时更新数值为0,省去第一轮计算
            new_index = self.troubled_nodes(network)
        elif k > max_k:
            # k 过大时,不进行更新,限制get_away的更新幅度(最大步长)
            new_index = np.array(False)
        else:
            # 更新
            up = network + k * step * head_dir
            down = network - k * step * head_dir
            good_up = ~self.troubled_nodes(up)
            good_down = ~self.troubled_nodes(down)
            network[good_down] = down[good_down]
            network[good_up] = up[good_up]

            new_index = ~(good_down | good_up)  # 仍然位于障碍物内的点

        # step 2 update the bad nodes
        if new_index.any():  # 存在处于障碍物内的点
            network[new_index] = self.get_away(
                network=network[new_index],
                step=step[new_index],
                head_dir=head_dir[new_index],
                k=k + 1,
                max_k=max_k,
            )

        return network
