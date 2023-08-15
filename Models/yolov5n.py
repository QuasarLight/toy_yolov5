import math
import torch
import torch.nn as nn
from Models.common import CBS, C3, SPPF, Concat
from Utils.torch_utils import initialize_weights

nc = 1
anchors = [[10,13, 16,30, 33,23], # P3/8
           [30, 61, 62, 45, 59, 119], # P4/16
           [116, 90, 156, 198, 373, 326]] # P5/32

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone1 = nn.Sequential(CBS(c1=3, c2=16, k=6, s=2, p=2),
                                      CBS(c1=16, c2=32, k=3, s=2, p=1),
                                      C3(c1=32, c2=32, n=1, shortcut=True),
                                      CBS(c1=32, c2=64, k=3, s=2, p=1),
                                      C3(c1=64, c2=64, n=2, shortcut=True))

        self.backbone2 = nn.Sequential(CBS(c1=64, c2=128, k=3, s=2, p=1),
                                      C3(c1=128, c2=128, n=3, shortcut=True))

        self.backbone3 = nn.Sequential(CBS(c1=128, c2=256, k=3, s=2, p=1),
                                      C3(c1=256, c2=256, n=1, shortcut=True),
                                      SPPF(c1=256, c2=256,k=5))

    def forward(self,x):
        y1 = self.backbone1(x)
        y2 = self.backbone2(y1)
        y3 = self.backbone3(y2)
        return [y1, y2, y3]

class Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.cbs1 = CBS(c1=256, c2=128, k=1, s=1, p=0)
        self.upsample = nn.Upsample(size=None, scale_factor=2, mode='nearest')
        self.concat = Concat(dimension=1)
        self.c3_1_1 = C3(c1=256, c2=128, n=1, shortcut=False)
        self.cbs2 = CBS(c1=128, c2=64, k=1, s=1, p=0)
        self.c3_1_2 = C3(c1=128, c2=64, n=1, shortcut=False)
        self.cbs3 = CBS(c1=64, c2=64, k=3, s=2, p=1)
        self.c3_1_3 = C3(c1=128, c2=128, n=1, shortcut=False)
        self.cbs4 = CBS(c1=128, c2=128, k=3, s=2, p=1)
        self.c3_1_4 = C3(c1=256, c2=256, n=1, shortcut=False)

    def forward(self,x):
        y1 = self.cbs1(x[2])
        y2 = self.cbs2(self.c3_1_1(self.concat([self.upsample(y1),x[1]])))
        y3 = self.c3_1_2(self.concat([self.upsample(y2),x[0]]))
        y4 = self.c3_1_3(self.concat([self.cbs3(y3),y2]))
        y5 = self.c3_1_4(self.concat([self.cbs4(y4),y1]))
        return [y3,y4,y5]

class Head(nn.Module):
    stride = None  # strides computed during build

    def __init__(self, nc=80, anchors = [], ch = []):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # nl为检测的特征图的数量
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.conv = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.conv[i](x[i])  # conv
            # x(bs,18,20,20) to x(bs,3,20,20,6)，ny为矩阵行数也就是图像的高，nx为矩阵列数也就是图像的宽
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            # inference
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                # 对原始特征图的所有数值进行sigmoid激活
                y = x[i].sigmoid()

                # 边框回归，注意边框回归方式与yolov3不同, y的shape为（bs，na，ny，nx，no）
                y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i] * self.stride[i]  # wh

                # z是一个列表分别存储三个特征图的经过边框回归后的检测结果，其中每一个特征图的shape变为（bs，na*ny*nx，no）
                z.append(y.view(bs, -1, self.no))

        # x为原始特征图，torch.cat(z, 1)为3个经过边框回归的特征图(shape又变了一下，不是原始的特征图的shape了)的按列拼接，shape为（bs，na*ny1*nx1+na*ny2*nx2+na*ny3*nx3，no）
        return x if self.training else torch.cat(z, 1)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        # grid存储每个格点的坐标，shape为（bs，na，ny，nx，2），最后一维为第几列第几行（方便后续边框回归）
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        # anchor_grid存储对应特征图的anchor的宽高，shape为（bs，na，ny，nx，2），数值单位是像素，最后一维第一个值为anchor的宽，第二个值为anchor的高
        anchor_grid = self.anchors[i].clone().view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid

class yolov5n(nn.Module):
    def __init__(self, input_ch=3, nc=nc, anchors=anchors):
        super().__init__()
        self.nc = nc
        self.anchors = anchors

        # 定义模型
        self.model = nn.Sequential(Backbone(),Neck(),Head(nc=self.nc, anchors = self.anchors, ch = [64,128,256]))

        # 构造 strides, anchors
        head = self.model[-1]  # Head()
        if isinstance(head, Head):
            s = 640  # 2x min stride
            head.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, input_ch, s, s))])  # forward
            head.anchors /= head.stride.view(-1, 1, 1)
            self.stride = head.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)

    def forward(self, x):
        return self.model(x)

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.conv, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

if __name__ == '__main__':
    # device
    device = torch.device('cuda')

    # Create Models
    model = yolov5n().to(device)
    # print(Models)

    # torch.save(Models, "yolov5n.pt")
    # x = torch.randn(1, 3, 640, 640).to(device)
    # script_model = torch.jit.trace(Models, x)
    # script_model.save("yolov5n.pt")

    model.train()
    # model.eval()

    # img = torch.rand(1, 3, 640, 640)
    # # print(img.type())
    # img = img.to(device)
    # # print(img.type())
    # pred = model(img)
    # print(pred[0].type())
    # print(pred[0].shape)

    # print(type(model.model[-1]))



