import numpy as np

k3 = np.array([1, 0, -1,
               2, 0, -2,
               1, 0, -1])

k5 = np.array([2, 1, 0, -1, -2,
               3, 2, 0, -2, -3,
               4, 3, 0, -3, -4,
               3, 2, 0, -2, -3,
               2, 1, 0, -1, -2])

k7 = np.array([3, 2, 1, 0, -1, -2, -3,
               4, 3, 2, 0, -2, -3, -4,
               5, 4, 3, 0, -3, -4, -5,
               6, 5, 4, 0, -4, -5, -6,
               5, 4, 3, 0, -3, -4, -5,
               4, 3, 2, 0, -2, -3, -4,
               3, 2, 1, 0, -1, -2, -3])

k9 = np.array([4, 3, 2, 1, 0, -1, -2, -3, -4,
               5, 4, 3, 2, 0, -2, -3, -4, -5,
               6, 5, 4, 3, 0, -3, -4, -5, -6,
               7, 6, 5, 4, 0, -4, -5, -6, -7,
               8, 7, 6, 5, 0, -5, -6, -7, -8,
               7, 6, 5, 4, 0, -4, -5, -6, -7,
               6, 5, 4, 3, 0, -3, -4, -5, -6,
               5, 4, 3, 2, 0, -2, -3, -4, -5,
               4, 3, 2, 1, 0, -1, -2, -3, -4])


def sobel_gain_factor(ksize):
    lut = {3: np.sum(np.abs(k3)),
           5: np.sum(np.abs(k5)),
           7: np.sum(np.abs(k7)),
           9: np.sum(np.abs(k9))}

    assert(ksize in lut)
    return lut[ksize]


if __name__ == '__main__':
    print("k3 gain factor", sobel_gain_factor(3))
    print("k5 gain factor", sobel_gain_factor(5))
    print("k7 gain factor", sobel_gain_factor(7))
    print("k9 gain factor", sobel_gain_factor(9))
