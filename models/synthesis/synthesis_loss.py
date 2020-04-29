from models.synthesis.losses import *
from models.synthesis.metrics import *

class SynthesisLoss(nn.Module):
    """
    Class for calculation of L1, content losses and PSNR, SSIM metrics at the same time.
    Losses to calculate should be specified.
    """
    def __init__(self, losses=['1.0_l1', '10.0_content']):
        """
        :param losses: 
            loss specification, str of the form: 'lambda_loss'
            lambda is used to weight different losses
            l1 and content/perceptual losses are summed if both are specified
            used in the forward method
        """

        super().__init__()

        lambdas, loss_names = zip(*[loss_name.split("_") for loss_name in losses]) # Parse lambda and loss_names from str
        print("Loss names:", loss_names)
        print("Weight of each loss:", lambdas)
        lambdas = [float(l) for l in lambdas] # [str] -> [float]

        loss_names += ("PSNR", "SSIM") # After this step, loss_names may include l1, content, PSNR, SSIM

        self.lambdas = lambdas
        self.losses = nn.ModuleList(
            [self.get_loss_from_name(loss_name) for loss_name in loss_names]
        )

    def get_loss_from_name(self, name):
        if name == "l1":
            loss = L1LossWrapper()
        elif name == "content":
            loss = PerceptualLoss()
        elif name == "PSNR":
            loss = PSNR()
        elif name == "SSIM":
            loss = SSIM()

        if torch.cuda.is_available():
            return loss.cuda()

    def forward(self, pred_img, gt_img):
        """
        For loss or metric function provided, evaluate the function of prediction and target.
        All results of individual functions along with the total loss returned in a dictionary.
        PSNR and SSIM are not added to total loss.
        
        :param pred_img: NVS image outputted from the generator
            used for loss calculation/metric evaluation
        :param gt_img: GT image for the novel view
            used for loss calculation/metric evaluation
        """

        # Initialize output dict
        loss_dir = {"Total Loss": 0}

        for i, func in enumerate(self.losses):
            # Evaluate each different loss(L1, Content/Perceptual)/metric(SSIM, PSNR) function between the prediction and target
            loss = func(pred_img, gt_img)

            # Consider only L1 and content/perceptual loss in the total loss calculation, i.e. do not add SSIM, PSNR metrics to total loss
            if "Total Loss" in loss.keys():
                loss_dir["Total Loss"] += loss["Total Loss"] * self.lambdas[i]
            
            # Merge both dicts and store the resulting dict in loss_dir
            loss_dir = dict(loss, **loss_dir)

        return loss_dir