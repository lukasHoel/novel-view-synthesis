# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from util.ssim import ssim
from torch.nn.functional import mse_loss

# The SSIM metric
def ssim_metric(img1, img2, mask=None):
    return ssim(img1, img2, mask=mask, size_average=False)


# The PSNR metric
def psnr(img1, img2):
    psnr = 10 * (1 / mse_loss(img1, img2)).log10()
    return psnr


# The perceptual similarity metric
def perceptual_sim(img1, img2, vgg16):
    # First extract features
    dist = vgg16(img1 * 2 - 1, img2 * 2 - 1)

    return dist
