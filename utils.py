import os
import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision.transforms import ToTensor


def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()[0]


def get_A(x):
    x_np = np.clip(torch_to_np(x), 0, 1)
    x_pil = np_to_pil(x_np)
    h, w = x_pil.size
    windows = (h + w) / 2
    A = x_pil.filter(ImageFilter.GaussianBlur(windows))
    A = ToTensor()(A)
    return A.unsqueeze(0)


if __name__ == '__main__':
    a = torch.randn(1, 3, 512, 512)
    b = get_A(a)
