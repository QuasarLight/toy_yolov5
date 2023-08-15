from pathlib import Path
import argparse
import torch
from copy import deepcopy
from Models.yolov5n import yolov5n
from multiprocessing import Pool, cpu_count
import time
from torch.utils.data import DataLoader

# file = Path(__file__).resolve()
# print(file)
# print(file.stem)

# known = False
# parser = argparse.ArgumentParser()
# parser.add_argument('--data', type=str, default='data/CF.yaml', help='dataset.yaml path')
# parser.add_argument('--name', default='cf_yolov5n_2', help='save to project/name')
# opt = parser.parse_known_args()[0] if known else parser.parse_args()
# print(opt)

# a = torch.ones(16,3,20,20,18)
# # print(a)
# print(a.shape)
# b = torch.tensor(a.shape)
# print(b)
# c = b[[3,2,3,2]]
# print(c)
# d = [b[3],b[2],b[3],b[2]]
# print(d)

# device = torch.device('cuda')
# # model = yolov5n().to(device)
# model = torch.load('F:\Object Detection\Simplified _Yolov5\Checkpoints\yolov5n.pth').to(device)
# model.eval()
# # torch.save(model, 'yolov5n_test1.pt')
#
# input=torch.rand(size=(1, 3, 640, 640)).to(device)
# model = torch.jit.trace(model, input)
# torch.jit.save(model, 'yolov5n_test2.pt')

def func1(x1):
    time.sleep(0.1)
    return (x1, x1 * x1)
def func2(x1, x2):
    time.sleep(0.1)
    return (x1, x2, x1 + x2)
if __name__ == "__main__":
    print('CPU核的数量：', cpu_count())
    begin = time.time()
    with Pool(8) as p:
        # for i in range(8):
        #     result = p.apply_async(func2, (1, i,))
        result = p.imap(func2, [(1, i) for i in range(8)])
        p.close()
        p.join()

    during = time.time() - begin

    print(during)


