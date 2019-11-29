import os
from matplotlib import pyplot as plt


def draw():
    log_dir = 'experiments'
    for log_path in os.listdir(log_dir):
        log = os.path.join(log_dir, log_path)
        with open(log, 'r') as f:
            epoch = []
            loss = []
            test_acc = []
            count = 0
            for line in f.readlines():
                data = line.split(' ')
                if data[0] == 'INFO:root:Epoch':
                    count += 1
                    epoch.append(count)
                    loss.append(float(data[3]))
                    test_acc.append(float(data[6][0:-2]))
        plt.figure(1)
        plt.plot(epoch, loss, label=log_path)
        plt.figure(2)
        plt.plot(epoch, test_acc, label=log_path)
    plt.figure(1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.figure(2)
    plt.xlabel('epoch')
    plt.ylabel('test_acc')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    draw()