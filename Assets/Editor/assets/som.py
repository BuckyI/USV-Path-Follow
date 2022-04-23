import numpy as np
import matplotlib.pyplot as plt
import random

from skimage import io, color, transform, morphology

from sklearn.cluster import SpectralClustering

"math operation function"


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
