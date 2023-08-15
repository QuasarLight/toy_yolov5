import os
import time
import torch
from tqdm import tqdm
from torch.cuda import amp
import torch.optim as optim
from Models.yolov5n import yolov5n
from Losses.loss import ComputeLoss
from torch.optim import lr_scheduler
from Utils.torch_utils import ModelEMA
from Datasets.datasets import create_dataloader

if __name__=='__main__':
    # device initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('The current device is:', device)

    # train_dataset and train_dataloader
    list_path = r'E:\DataSets\Object Detection DataSets\CFData\TrainingData\train.txt'
    dataloader, dataset = create_dataloader(list_path=list_path,
                                            imgsz=640,
                                            batch_size=48,
                                            augment=True,
                                            workers=0,
                                            shuffle=True)

    # model
    model = yolov5n().to(device)
    model.train()

    # loss function
    compute_loss = ComputeLoss(model)

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.937, nesterov = True, weight_decay = 0.0005)
    # optimizer = optim.Adam(model.parameters(), lr = 0.01, betas=(0.937, 0.999),weight_decay = 0.0005)

    # learning rate scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = [100,200,250], gamma = 0.1)

    # model ema
    ema = ModelEMA(model)

    # training network
    scaler = amp.GradScaler(enabled=True)
    for epoch in range(300):
        print('Training Epoch: '+str(epoch))
        time.sleep(0.5)
        pbar = tqdm(dataloader,total=dataset.nb)
        for images, labels, _ in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with amp.autocast(enabled=True):
                pred_targets = model(images)
                loss, loss_items = compute_loss(pred_targets, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update(model)
            pbar.set_postfix(lbox=loss_items[0], lobj=loss_items[1], lcls=loss_items[2])
        time.sleep(0.5)
        scheduler.step()
        print('lbox:', loss_items[0], 'lobj:', loss_items[1], 'lcls:', loss_items[2])

    # save model
    save_dir = 'F:\Object Detection\Simplified _Yolov5\Checkpoints'
    print('Saving model...')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    torch.save(model.half(), os.path.join(save_dir, 'yolov5n_v1.pt'))
