import os
import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

MODEL_PATH = './saved_models/u2net/u2net.pth'
INPUT_DIR = 'test_assets'
OUTPUT_DIR = 'output'

def save_output(image_name, pred, d_dir):

    img = Image.fromarray(pred.squeeze().cpu().data.numpy() * 255).convert('RGB')

    img_src = np.array(Image.open(image_name))
    x, y = img_src.shape[1], img_src.shape[0]

    img_out = img.resize((x, y), resample=Image.BILINEAR)
    save_path = os.path.join(d_dir, os.path.splitext(os.path.basename(image_name))[0] + '.png')
    img_out.save(save_path)
    
    return save_path


def initialize_model(model_path = MODEL_PATH):
    if model_path.split('/')[-1] == 'u2net.pth':
        print("...Loading U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif model_path.split('/')[-1] == 'u2netp.pth':
        print("...Loading U2NEP---4.7 MB")
        net = U2NETP(3, 1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location='cpu'))

    net.eval()
    return net


def get_batch_saliency(net, image_dir=INPUT_DIR, output_dir=OUTPUT_DIR):

    print(image_dir)
    img_name_list = glob.glob(image_dir + os.sep + '*')

    test_salobj_dataset = SalObjDataset(
        img_name_list=img_name_list,
        lbl_name_list=[],
        transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
    )
    test_salobj_dataloader = DataLoader(
        test_salobj_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    ret_path = []

    with torch.no_grad():
        for i_test, data_test in enumerate(test_salobj_dataloader):

            print("Inferencing:", img_name_list[i_test].split(os.sep)[-1])

            inputs_test = data_test['image']
            inputs_test = inputs_test.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_test = inputs_test.cuda()

            pred = net(inputs_test)[:, 0, :, :]

            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            save_path = save_output(img_name_list[i_test], pred, output_dir)
            del pred
            
            ret_path.append(os.path.abspath(save_path))

    return ret_path


def get_single_saliency(net, image_path, output_dir=OUTPUT_DIR):

    img_name_list = [image_path]

    test_salobj_dataset = SalObjDataset(
        img_name_list=img_name_list,
        lbl_name_list=[],
        transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
    )
    test_salobj_dataloader = DataLoader(
        test_salobj_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    with torch.no_grad():
        for i_test, data_test in enumerate(test_salobj_dataloader):

            print("Inferencing:", img_name_list[i_test].split(os.sep)[-1])

            inputs_test = data_test['image']
            inputs_test = inputs_test.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_test = inputs_test.cuda()

            pred = net(inputs_test)[:, 0, :, :]

            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            save_path = save_output(img_name_list[i_test], pred, output_dir)
            del pred
            
            return os.path.abspath(save_path)


if __name__ == "__main__":
    net = initialize_model()

    print("------- Batch Test -------")
    ret = get_batch_saliency(net)
    print(ret)

    print("------- Single Test -------")
    ret_2 = get_single_saliency(net, './test_assets/pixelmator.jpg')
    print(ret_2)
    