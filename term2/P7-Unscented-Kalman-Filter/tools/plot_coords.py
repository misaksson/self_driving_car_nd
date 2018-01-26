import numpy as np
import matplotlib.pyplot as plt


def load_coords(file_name):

    xs, ys, xds, yds = [], [], [], []
    with open(file_name) as fid:
        for line in fid:
            coords = np.array(line.split()).astype(np.float)
            if len(coords) == 2:
                x, y = coords
            else:
                x, y, xd, yd = coords
                xds.append(xd)
                yds.append(yd)
            xs.append(x)
            ys.append(y)

    return xs, ys, xds, yds


def plot_coords(dataset):
    plt.figure(figsize=(15, 10))
    xs, ys, xds, yds = load_coords('../ground_truth' + dataset + '.txt')
    plt.plot(xs, ys, 'k', label='Actual', linewidth=4)
    xs, ys, xds, yds = load_coords('../estimations' + dataset + '.txt')
    plt.plot(xs, ys, 'lime', label='Estimated', linewidth=2)
    xs, ys, _, _ = load_coords('../radar' + dataset + '.txt')
    plt.plot(xs, ys, 'bo', label='Radar', markersize=1)
    xs, ys, _, _ = load_coords('../lidar' + dataset + '.txt')
    plt.plot(xs, ys, 'ro', label='Lidar', markersize=1)
    plt.title('Dataset' + dataset)
    plt.legend(loc='upper right')


plot_coords(dataset='1')
plot_coords(dataset='2')
plt.show()
