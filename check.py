import math
import numpy as np


def check(cntk_path, pytorch_path):
    eps = 1e-4

    print('Comparing...')

    cntk_data = np.loadtxt(cntk_path)
    pytorch_data = np.loadtxt(pytorch_path)
    if cntk_data.shape != pytorch_data.shape:
        cntk_data = np.transpose(cntk_data)
    assert cntk_data.shape == pytorch_data.shape
    cntk_data = cntk_data.flatten()
    pytorch_data = pytorch_data.flatten()
    cnt = len(cntk_data)
    diff = 0.0
    for i in range(cnt):
        diff += math.fabs(cntk_data[i] - pytorch_data[i])
    print('Difference: %.10f, totally %d.' % (diff, cnt))

    # cntk_data = []
    # with open(cntk_path, 'r') as f:
    #     for line in f.readlines():
    #         data = line.split(' ')
    #         for i in range(len(data)):
    #             cntk_data.append(float(data[i]))
    #     cnt = len(cntk_data)
    #     print('Read CNTK data, totally %d.' % cnt)
    #
    # pytorch_data = []
    # with open(pytorch_path, 'r') as f:
    #     for line in f.readlines():
    #         data = line.split(' ')
    #         for i in range(len(data)):
    #             try:
    #                 pytorch_data.append(float(data[i]))
    #             except:
    #                 pass
    #     cnt = len(pytorch_data)
    #     print('Read PyTorch data, totally %d.' % cnt)
    #
    # if len(cntk_data) != len(pytorch_data):
    #     raise RuntimeError('Different number between CNTK and PyTorch.')
    #
    # cnt = len(cntk_data)
    # ans = 0.0
    # for i in range(cnt):
    #     ans += math.fabs(cntk_data[i] - pytorch_data[i])
    #     # if math.fabs(cntk_data[i] - pytorch_data[i]) < eps:
    #     #     same += 1
    #     # else:
    #     #     print('CNTK: %f vs PyTorch: %f', cntk_data[i], pytorch_data[i])
    # # print('Same: %d' % same)
    # print('Difference: %f' % ans)


# convOutput_cntk = './Compare/Grad_3_144_cntk.txt'
# convOutput_pytorch = './Compare/Grad_3_144_pytorch.txt'
convOutput_cntk = './Compare/fcOutput_100_2_cntk.txt'
convOutput_pytorch = './Compare/fcOutput_100_2_pytorch.txt'

check(convOutput_cntk, convOutput_pytorch)
