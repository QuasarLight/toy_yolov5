import torch
import torchvision

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

# nms，不支持batch nms
def nms(prediction, conf_thres=0.25, iou_thres=0.45):
    nc = prediction.shape[2] - 5  # 类别数
    xc = prediction[..., 4] > conf_thres  # obj概率大于阈值的box的坐标（用于过滤掉obj概率小于阈值的box）

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height

    output = torch.zeros((0, 6), device=prediction.device)

    x = prediction[0]
    x = x[xc[0]]  # 图片中满足obj概率大于阈值的box,输出x的shape为n*no（比如10*6），n为obj概率大于阈值的box个数，no为4+1+c

    # 如果该图像没有满足条件的检测框，则返回
    if not x.shape[0]:
        return output

    # 计算置信度
    x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

    # Box (center x, center y, width, height) to (x1, y1, x2, y2)
    box = xywh2xyxy(x[:, :4])

    # Detections matrix nx6 (xyxy, conf, cls)
    conf, j = x[:, 5:].max(1, keepdim=True)
    x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres] #只保留置信度大于于阈值的box

    # 再检查一边
    n = x.shape[0]  # number of boxes
    if not n:  # no boxes
        return output

    # NMS
    c = x[:, 5:6] * max_wh  # classes
    boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores +c目的是区分不同类别的检测框（牛逼）
    i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
    output = x[i]

    # output的shape为n*6，n为满足条件的box数，6为(xyxy, conf, cls)
    return output