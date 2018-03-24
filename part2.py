# Part 2 "batch_size"
import numpy as np
batch_size = 10


def create_batch():
    x = np.random.rand(batch_size, 2)*10
    y = np.sin(x[:, 0]) + np.sin(x[:, 1])
    y = y.reshape(-1, 1)
    return(x, y)


print(create_batch())
