import numpy as np
import math


def softmax(z):
    if len(z.shape) == 1:
        return np.exp(z) / np.sum(np.exp(z), axis=0)
    elif len(z.shape) == 2:
        res = []
        print(z.T)
        for i in z.T:
            res.append(np.exp(i) / np.sum(np.exp(i), axis=0))
        res = np.array(res)
        return res.T


logits3 = np.array([
    [1, 2, 3, 6],
    [2, 4, 5, 6],
    [3, 8, 7, 6]])

logits = np.array([3.0, 1.0, 0.2])
print(softmax(logits3))
