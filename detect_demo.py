import cv2
import torch
import numpy as np
from Utils.nms import nms

# 读取图片和预处理
device = torch.device('cuda')
img_path = 'img1.png'
img0 = cv2.imread(img_path)
img = img0
img = img.astype(np.float32)
img /= 255.0
img = img.transpose(2, 0, 1)
img = np.expand_dims(img, axis=0)
img = torch.from_numpy(img).to(device)

# 模型载入
model_path = 'Checkpoints\yolov5n.pth'
model = torch.load(model_path, map_location = device)
model.eval()

# 模型推理
pred = model(img)

# NMS
conf_threshold = 0.4
iou_threshold = 0.45
pred = nms(pred, conf_threshold, iou_threshold)

# 对最终检测结果进行处理和利用
if len(pred):
    for i in range(len(pred)):
        xyxy = pred[i][0:4]
        cv2.rectangle(img0,  # 图片
                     (int(xyxy[0]), int(xyxy[1])),  # (xmin, ymin)左上角坐标
                     (int(xyxy[2]), int(xyxy[3])),  # (xmax, ymax)右下角坐标
                     (0, 0, 255), 2)  # 颜色，线条宽度
        cv2.putText(img0, 'head', (int(xyxy[0]) - 3, int(xyxy[1]) - 2), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
cv2.imshow("detection_results", img0)
cv2.waitKey(0)