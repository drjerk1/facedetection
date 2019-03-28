import math
import numpy as np

def r(x):
    return int(round(x))

def sigmoid(x):
    return 1 / (np.exp(-x) + 1)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1).reshape((x.shape[0], 1)))
    return e_x / np.sum(e_x, axis=-1).reshape((x.shape[0], 1))

def cosine_dist_norm(a, b):
    return 1 - np.sum(a * b, axis=-1)
