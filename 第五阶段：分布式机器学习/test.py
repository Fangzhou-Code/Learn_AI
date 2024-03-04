# Pytorch 点对点通信（同步和异步）
'''同步'''


import torch
from setuptools._distutils import dist


def run(rank, size):
    tensor = torch.zeros(1)
    if rank==0:
        tensor += 1
        # send tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank',rank, 'has data ', tensor[0])
