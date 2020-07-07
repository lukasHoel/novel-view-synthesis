import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class LPRegionLoss(nn.Module):

    def __init__(self, p=1, lower_weight = 0.0, higher_weight = 2.0):
        super().__init__()
        self.p = int(p)
        self.lower_weight = lower_weight
        self.higher_weight = higher_weight

    def forward(self, pred_img, gt_img, mask_lower_region, mask_higher_region):

        # check if lower and higher region overlap. If so: truncate the lower region
        overlap = mask_lower_region == mask_higher_region
        mask_lower_region[overlap] = False

        # apply lower_weight to the lower_important region
        pred_img[:, mask_lower_region] *= self.lower_weight
        gt_img[:, mask_lower_region] *= self.lower_weight

        # apply higher_weight to the higher_important region
        pred_img[:, mask_higher_region] *= self.higher_weight
        gt_img[:, mask_higher_region] *= self.higher_weight

        # calculate lp loss
        loss = torch.mean((pred_img - gt_img)**self.p)
        return {"LPRegion": loss, "Total Loss": loss}



class L1LossWrapper(nn.Module):
    """Wrapper of the L1Loss so that the format matches what is expected"""
    def forward(self, pred_img, gt_img):
        """
        :param pred_img: NVS image outputted from the generator
            used for loss calculation/metric evaluation
        :param gt_img: GT image for the novel view
            used for loss calculation/metric evaluation
        """
        err = F.l1_loss(pred_img, gt_img)
        return {"L1": err, "Total Loss": err}

class VGG19(nn.Module):
    """Pretrained VGG19 architecture to be utilized in PerceptualLoss"""
    def __init__(self, requires_grad=False):
        super().__init__()
        # Using "features" part of VGG19 only, discarding avgpool and classifier parts
        vgg_pretrained_features = vgg19(pretrained=True).features
        # Slices are generated to keep intermediary results in the forward pass
        self.slice1 = nn.Sequential() # Layers: 0-1
        self.slice2 = nn.Sequential() # Layers: 2-6
        self.slice3 = nn.Sequential() # Layers: 7-11
        self.slice4 = nn.Sequential() # Layers: 12-20
        self.slice5 = nn.Sequential() # Layers: 21-29
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
        h_relu1 = self.slice1(X)       # Output after layers 0-1
        h_relu2 = self.slice2(h_relu1) # Output after layers: 2-6
        h_relu3 = self.slice3(h_relu2) # Output after layers: 7-11
        h_relu4 = self.slice4(h_relu3) # Output after layers: 12-20
        h_relu5 = self.slice5(h_relu4) # Output after layers: 21-29

        # Intermediate results from different slices of layers are stored
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class PerceptualLoss(nn.Module):
    """
    Implementation of the perceptual/content loss to enforce that a generated image matches the given image.
    Pretrained VGG architecture is used for the perceptual/content loss
    Adapted from SPADE's implementation
    (https://github.com/NVlabs/SPADE/blob/master/models/networks/loss.py)
    """
    def __init__(self):
        super().__init__()
        self.model = VGG19(requires_grad=False) # Freeze the network
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0] # Weights used for the contribution of the output of each slice to the total loss

    def forward(self, pred_img, gt_img):
        """
        :param pred_img: NVS image outputted from the generator
            used for loss calculation/metric evaluation
        :param gt_img: GT image for the novel view
            used for loss calculation/metric evaluation
        """
        gt_fs = self.model(gt_img)
        pred_fs = self.model(pred_img)

        # Calculate the losses for every output from each slice and sum them up (need unsqueeze in order to concatenate these together)
        loss = 0
        for i in range(0, len(gt_fs)):
            loss += self.weights[i] * self.criterion(pred_fs[i], gt_fs[i])

        return {"Perceptual": loss, "Total Loss": loss}