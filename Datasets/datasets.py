import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from Models.yolov5n import yolov5n
from Losses.loss import ComputeLoss
from multiprocessing.pool import Pool
from Utils.general import xywhn2xyxy, xyxy2xywhn
from torch.utils.data import Dataset, dataloader
from Utils.augmentations import Albumentations, random_perspective, letterbox, augment_hsv

NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of multiprocessing threads

# create dataloader
def create_dataloader(list_path, imgsz, batch_size, augment=True, workers=0, shuffle=True):
    dataset = LoadImagesAndLabels(list_path, imgsz, batch_size, augment=augment)

    return InfiniteDataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=workers,
                              pin_memory=True,
                              collate_fn = LoadImagesAndLabels.collate_fn), dataset

# LoadImagesAndLabels
class LoadImagesAndLabels(Dataset):

    def __init__(self, list_path, img_size=640, batch_size=16, augment=False):
        self.img_size = img_size
        self.augment = augment
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.list_path = list_path
        self.albumentations = Albumentations() if augment else None

        f = []  # image files
        list_path = Path(list_path)  # os-agnostic
        with open(list_path) as list_file:
            img_paths = list_file.read().strip().splitlines()
            parent = str(list_path.parent) + os.sep
            f += [img_path for img_path in img_paths]
        self.img_files = f

        self.label_files = img2label_paths(self.img_files)  # labels

        # cache label
        cache = self.cache_labels()  # cache labels

        # Read cache
        nf, nm, ne, nc, nt = cache.pop('results')  # found, missing, empty, corrupted, total
        msgs = cache.pop('msgs')
        labels, shapes = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        ni = len(shapes)  # number of images
        bi = np.floor(np.arange(ni) / batch_size).astype(np.int)  # batch index
        self.nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.ni = ni # number of images
        self.indices = range(ni)

    def cache_labels(self):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        print("Scanning images and labels...")
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(verify_image_label, zip(self.img_files, self.label_files)), total=len(self.img_files))
            for im_file, l, shape, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [l, shape]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        pbar.close()

        x['results'] = nf, nm, ne, nc, len(self.img_files)
        x['msgs'] = msgs  # warnings

        return x

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        if self.augment:
            # Load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None
        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])


        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4)

            # Flip left-right
            if random.random() < 0.5:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = img.astype(np.float32)
        img /= 255.0
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, shapes

    @staticmethod
    def collate_fn(batch):
        img, label, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), shapes

def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

def verify_image_label(args):
    # Verify one image-label pair
    im_file, lb_file = args
    nm, nf, ne, nc, msg= 0, 0, 0, 0, ''  # number (missing, found, empty, corrupt), message
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = im.size  # image size

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                l = np.array(l, dtype=np.float32)
            nl = len(l)
            if nl:
                assert l.shape[1] == 5, f'labels require 5 columns, {l.shape[1]} columns detected'
                assert (l >= 0).all(), f'negative label values {l[l < 0]}'
                assert (l[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {l[:, 1:][l[:, 1:] > 1]}'
                _, i = np.unique(l, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    l = l[i]  # remove duplicates
                    msg = f'{prefix}WARNING: {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1  # label empty
                l = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            l = np.zeros((0, 5), dtype=np.float32)
        return im_file, l, shape, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'WARNING: {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, nm, nf, ne, nc, msg]

def load_mosaic(self, index):
    labels4 = []
    s = self.img_size
    yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    random.shuffle(indices)
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels= self.labels[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
        labels4.append(labels)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in labels4[:, 1:]:
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()

    # Augment
    img4, labels4 = random_perspective(im = img4,
                                       targets = labels4,
                                       degrees=0.0,
                                       translate=0.1,
                                       scale=0.5,
                                       shear=0.0,
                                       perspective=0.0,
                                       border=self.mosaic_border)  # border to remove

    return img4, labels4

def load_image(self, i):
    # loads 1 image from dataset index 'i', returns im, original hw, resized hw
    path = self.img_files[i]
    im = cv2.imread(path)  # BGR
    assert im is not None, f'Image Not Found {path}'
    h0, w0 = im.shape[:2]  # orig hw
    r = self.img_size / max(h0, w0)  # ratio
    if r != 1:  # if sizes are not equal
        im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                        interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
    return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized

# InfiniteDataLoader
class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

if __name__ == '__main__':
    list_path = r'E:\DataSets\Object Detection DataSets\CFData\TrainingData\train.txt'
    trainloader, dataset = create_dataloader(list_path = list_path,
                                             imgsz = 640,
                                             batch_size = 16,
                                             augment = True,
                                             workers = 0,
                                             shuffle = True)
    # print('number of batches:',dataset.nb)

    device = torch.device('cuda')
    model = yolov5n().to(device)
    model.train()

    compute_loss = ComputeLoss(model)

    # print(len(trainloader))
    for images, labels, shape in trainloader:
        # print(type(images))
        # print(images.shape)
        # print(type(labels))
        # print(labels.shape)
        # print(shape)
        # print(images.type())
        # print(labels.type())
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        # print(images.type())
        pred_targets = model(images)

        loss, loss_items = compute_loss(pred_targets, labels)
        print('lbox:',loss_items[0],'lobj:',loss_items[1],'lcls:',loss_items[2])
        # print('loss_items:',loss_items)














