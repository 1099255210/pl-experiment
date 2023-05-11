# -*- coding: utf-8 -*-
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import canvas
from model import generator
from PIL import Image, ImageDraw

from collections import OrderedDict
import numpy as np

import gpuinfo

LAYOUT_WIDTH = 513
LAYOUT_HEIGHT = 750
LAYOUT_SIZE = (LAYOUT_WIDTH, LAYOUT_HEIGHT)

TRANSFORM = transforms.Compose([
    transforms.Resize([350, 240]),
    transforms.ToTensor()
])

torch.manual_seed(0)
gpu = torch.cuda.is_available()
device_ids = gpuinfo.gpu_available
# print(device_ids)
device = torch.device(f"cuda:{device_ids[0]}" if gpu else "cpu")

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

 
def random_init(batch, max_elem):
    coef = [0.1, 0.8, 1, 1]
    cls_1 = torch.tensor(
        np.random.choice(4, size=(batch, max_elem, 1), p=np.array(coef) / sum(coef)), dtype=torch.int64
    )
    cls = torch.zeros((batch, max_elem, 4))
    cls.scatter_(-1, cls_1, 1)
    box_xyxy = torch.normal(0.5, 0.15, size=(batch, max_elem, 1, 4))
    box = box_xyxy_to_cxcywh(box_xyxy)
    init_layout = torch.concat([cls.unsqueeze(2), box], dim=2)
    return init_layout


def remove_invalid_box(img_size, cls, box):
    w, h = img_size
    
    for i, b in enumerate(box):
        xl, yl, xr, yr = b
        if xl < 0 or yl < 0:
            cls[i] = 0
            box[i] = [0, 0, 0, 0]
            continue
        xl = max(0, xl)
        yl = max(0, yl)
        xr = min(w, xr)
        yr = min(h, yr)
        if cls[i]:
            if abs((xr - xl) * (yr - yl)) < (w / 100) * (h / 100) * 10:
                cls[i] = 0
                box[i] = [0, 0, 0, 0]

    return cls, box


def draw_box(img, elems):
    draw = ImageDraw.ImageDraw(img)
    cls_color_dict = {1: 'green', 2: 'red', 3: 'orange'}
    elems = sorted(list(filter(lambda x: x[0][0], elems)), key=lambda x: x[0], reverse=True)

    for cls, box in elems:
        print(cls, box)
        draw.rectangle(tuple(box), fill=None, outline=cls_color_dict[cls[0]], width=5)

    img = Image.alpha_composite(img, img.convert("RGBA").point(lambda p: p * 0.3))
    return img


def get_result(G, img_list, noise_layout):
    G.eval()
    clses = []
    boxes = []
    with torch.no_grad():
        for img in img_list:
            img = img.unsqueeze(0)
            img = img.to(device)
            cls, box = G(img, noise_layout.to(device))
            cls = torch.argmax(cls.detach().cpu(), dim=-1, keepdim=True).numpy()
            box = box_cxcywh_to_xyxy(box.detach().cpu()).numpy()
            cls = cls[0]
            box = box[0]
            box[:, 1::2] *= LAYOUT_HEIGHT
            box[:, ::2] *= LAYOUT_WIDTH
            cls = cls.astype(int)
            box = box.astype(int)
            clses.append(cls)
            boxes.append(box)
    return clses, boxes
    
def get_single_layout(
        image_path:str='test_assets/img/3.png',
        sal_path:str='test_assets/sal/3.png'
    ):

    max_elem = 32
    args_g = {
        "backbone": "resnet50",
        "in_channels": 8,
        "out_channels": 32,
        "hidden_size": max_elem * 8,
        "num_layers": 4,
        "output_size": 8,
        "max_elem": max_elem
    }
    G = generator(args_g)
    
    ckpt_path = "output/DS-GAN-Epoch300.pth"
    ckpt = torch.load(ckpt_path)
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k[7:]
        new_state_dict[name] = v
    G.load_state_dict(new_state_dict)

    if gpu:
        G = G.to(device)

    noise_layout = random_init(1, max_elem)

    # noise_layout[:, :, :, :] *= 0
    # print(noise_layout)

    img_bg = TRANSFORM(Image.open(image_path).convert('RGB'))
    img_sal = TRANSFORM(Image.open(sal_path).convert('L'))
    img_in = [torch.concat([img_bg, img_sal])]

    clses, boxes = get_result(G, img_in, noise_layout)
    cls, box = remove_invalid_box(LAYOUT_SIZE, clses[0], boxes[0])

    img = Image.new('RGBA', LAYOUT_SIZE, (255, 255, 255))
    img = draw_box(img, zip(cls, box))
    img.save('layout.png')

def get_batch_layout(
        image_dir:str='test_assets/img',
        sal_dir:str='test_assets/sal'
    ):
    '''
    Not completed.
    '''

    test_bg_path = "test_assets/img"
    test_sal_dir = "test_assets/sal"
    test_batch_size = 1
    
    testing_set = canvas(test_bg_path, test_sal_dir, train=False)
    testing_dl = DataLoader(testing_set, num_workers=16, batch_size=test_batch_size, shuffle=False)

    print(f"testing_set: {len(testing_set)}")

    max_elem = 32
    args_g = {
        "backbone": "resnet50",
        "in_channels": 8,
        "out_channels": 32,
        "hidden_size": max_elem * 8,
        "num_layers": 4,
        "output_size": 8,
        "max_elem": max_elem
    }
    G = generator(args_g)
    
    ckpt_path = "output/DS-GAN-Epoch300.pth"
    ckpt = torch.load(ckpt_path)
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k[7:]
        new_state_dict[name] = v
    G.load_state_dict(new_state_dict)

    if gpu:
        G = G.to(device)

    noise_layout = random_init(test_batch_size, max_elem)
    clses, boxes = get_result(G, testing_dl, noise_layout)
    for i, (cls, box) in enumerate(zip(clses, boxes)):
        cls, box = remove_invalid_box(LAYOUT_SIZE, cls, box)
        

    # img = Image.new('RGBA', LAYOUT_SIZE, (255, 255, 255))
    # img = draw_box(img, zip(cls, box))
    # img.save('layout.png')


def test_inference():
    import u2net.inference as u2

    salnet = u2.initialize_model()

    img_path = 'test_assets/img/3.png'
    out_dir = 'test_assets/sal_u2'

    ret_path = u2.get_single_saliency(salnet, img_path, out_dir)

    print(ret_path)

    get_single_layout(image_path=img_path, sal_path=ret_path)


def test_random_init():
    nl = random_init(1, 32)
    nl[:, :, 1, ::2] *= LAYOUT_WIDTH
    nl[:, :, 1, 1::2] *= LAYOUT_HEIGHT
    nl = nl.numpy().astype(int)
    print(nl)

if __name__ == "__main__":
    test_inference()
    
    # test_random_init()

    ...