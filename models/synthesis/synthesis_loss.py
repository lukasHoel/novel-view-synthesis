from models.synthesis.losses import *
from models.synthesis.metrics import *

class SynthesisLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        # Loss inputs by --losses flag are in the format: 'lambda_loss', e.g.
        # - '1.0_l1'
        # - '10.0_content' 
        # Lambda is used to weight different losses
        print("Lambda-Loss pairs:", list(zip(*[l.split("_") for l in opt.losses])))
        lambdas, loss_names = zip(*[loss_name.split("_") for loss_name in opt.losses])
        lambdas = [float(l) for l in lambdas] # [str] -> [float]

        loss_names += ("PSNR", "SSIM") # NOTE: Loss names can include L1, Content, PSNR, SSIM

        self.lambdas = lambdas
        self.losses = nn.ModuleList(
            [self.get_loss_from_name(loss_name) for loss_name in loss_names]
        )

    def get_loss_from_name(self, name):
        if name == "l1":
            loss = L1LossWrapper()
        elif name == "content":
            loss = PerceptualLoss(self.opt)
        elif name == "PSNR":
            loss = PSNR()
        elif name == "SSIM":
            loss = SSIM()

        if torch.cuda.is_available():
            return loss.cuda()

    # TODO: To be removed
    def _forward(self, pred_img, gt_img):
        # Evaluate each different loss between the target and prediction and store results in a list
        losses = [loss(pred_img, gt_img) for loss in self.losses]

        loss_dir = {}
        for i, loss in enumerate(losses):
            # Consider only L1 and perceptual loss in the total loss calculation, i.e. do not add SSIM, PSNR values to total loss
            if "Total Loss" in loss.keys():
                if "Total Loss" in loss_dir.keys():
                    # Take each loss value and combine them with corresponding weights, accumulate on loss_dir["Total Loss"]
                    loss_dir["Total Loss"] = loss_dir["Total Loss"] + loss["Total Loss"] * self.lambdas[i]
                # Initialize loss_dir dict 
                else:
                    loss_dir["Total Loss"] = loss["Total Loss"]

            # Merge both dicts and store the resulting dict in loss_dir
            loss_dir = dict(loss, **loss_dir)

        return loss_dir

    def forward(self, pred_img, gt_img):
        # TODO: Check inputs: 
        # pred_img is the NVS image: input_img = batch["images"][0] @z_buffermodel.py: ZbufferModelPts.forward
        # What is ground truth specifically? output_img = batch["images"][-1] @z_buffermodel.py: ZbufferModelPts.forward

        # Initialize output dict
        loss_dir = {"Total Loss": 0}

        for i, func in enumerate(self.losses):
            # Evaluate each different loss(L1, Content)/metric(SSIM, PSNR) function between the target and prediction
            loss = func(pred_img, gt_img)

            # Consider only L1 and perceptual loss in the total loss calculation, i.e. do not add SSIM, PSNR metrics to total loss
            if "Total Loss" in loss.keys():
                loss_dir["Total Loss"] += loss["Total Loss"] * self.lambdas[i]
            
            # Merge both dicts and store the resulting dict in loss_dir
            loss_dir = dict(loss, **loss_dir)

        return loss_dir