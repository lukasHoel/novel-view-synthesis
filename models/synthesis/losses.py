import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class L1LossWrapper(nn.Module):
    """Wrapper of the L1Loss so that the format matches what is expected"""
    # TODO: To be removed
    def _forward(self, pred_img, gt_img):
        err = nn.L1Loss()(pred_img, gt_img)
        return {"L1": err, "Total Loss": err}

    def forward(self, pred_img, gt_img):
        err = F.l1_loss(pred_img, gt_img)
        return {"L1": err, "Total Loss": err}

class VGG19(nn.Module):
    """Pretrained VGG19 architecture to be utilized in PerceptualLoss"""
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        # Normalize the image so that it is in the appropriate range
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class PerceptualLoss(nn.Module):
    """
    Implementation of the perceptual loss to enforce that a generated image matches the given image.
    Pretrained VGG architecture is used for the perceptual loss
    Adapted from SPADE's implementation
    (https://github.com/NVlabs/SPADE/blob/master/models/networks/loss.py)
    """
    def __init__(self, opt):
        super().__init__()
        self.model = VGG19(requires_grad=False) # Freeze the network
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, pred_img, gt_img):
        gt_fs = self.model(gt_img)
        pred_fs = self.model(pred_img)

        # Collect the losses at multiple layers (need unsqueeze in order to concatenate these together)
        loss = 0
        for i in range(0, len(gt_fs)):
            loss += self.weights[i] * self.criterion(pred_fs[i], gt_fs[i])

        return {"Perceptual": loss, "Total Loss": loss}