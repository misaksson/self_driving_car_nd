import matplotlib.pyplot as plt


def plotFile(file_name, title, chi95=None):

    data = []
    with open(file_name) as fid:
        for line in fid:
            data.append(float(line))
    plt.plot(data)
    plt.title(title)
    if chi95 is not None:
        plt.plot([0, len(data) - 1], [chi95, chi95], 'r', label='chi-squared 95%')
    x1, x2, _, _ = plt.axis()
    plt.axis([x1, x2, 0, 10])


plt.figure(figsize=(8, 3))
plt.suptitle('Dataset 1')
plt.subplot(1, 2, 1)
plotFile('../NIS1_lidar.txt', title='LIDAR', chi95=5.991)
plt.ylabel('Normalized Innovation Squared')
plt.subplot(1, 2, 2)
plotFile('../NIS1_radar.txt', title='RADAR', chi95=7.815)
plt.legend(loc='lower right')

plt.figure(figsize=(8, 3))
plt.suptitle('Dataset 2')
plt.subplot(1, 2, 1)
plotFile('../NIS2_lidar.txt', title='LIDAR', chi95=5.991)
plt.ylabel('Normalized Innovation Squared')
plt.subplot(1, 2, 2)
plotFile('../NIS2_radar.txt', title='RADAR', chi95=7.815)
plt.legend(loc='lower right')
plt.show()
