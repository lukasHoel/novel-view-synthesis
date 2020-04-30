from models.synthesis.losses import *
from models.synthesis.metrics import *

class SynthesisLoss(nn.Module):
    """
    Class for simultaneous calculation of L1, content/perceptual losses.
    Losses to use should be passed as argument.
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

        self.lambdas = lambdas
        self.losses = nn.ModuleList(
            [self.get_loss_from_name(loss_name) for loss_name in loss_names]
        )

    def get_loss_from_name(self, name):
        if name == "l1":
            loss = L1LossWrapper()
        elif name == "content":
            loss = PerceptualLoss()
        else:
            raise "Invalid loss name in SynthesisLoss"
        # TODO: If needed, more loss classes can be introduced here later on.

        if torch.cuda.is_available():
            return loss.cuda()

    def forward(self, pred_img, gt_img):
        """
        For each loss function provided, evaluate the function with prediction and target.
        Results of individual functions along with the total loss returned in a dictionary.
        
        :param pred_img: NVS image outputted from the generator
            used for loss calculation/metric evaluation
        :param gt_img: GT image for the novel view
            used for loss calculation/metric evaluation
        """

        # Initialize output dict
        results = {"Total Loss": 0}

        for i, func in enumerate(self.losses):
            # Evaluate each different loss (L1, Content/Perceptual) function with the prediction and target
            out = func(pred_img, gt_img)

            # Add the contribution by each loss to the total loss wrt their weights (lambda)
            results["Total Loss"] += out["Total Loss"] * self.lambdas[i]
            
            # Merge both dicts and store the resulting dict in results
            results = dict(out, **results) 

        return results # Contains individual losses and weighted sum of these

class QualityMetrics(nn.Module):
    """
    Class for simultaneous calculation of known image quality metrics PSNR, SSIM.
    Metrics to use should be passed as argument.
    """
    def __init__(self, metrics=["PSNR", "SSIM"]):
        super().__init__()

        print("Metric names:", *metrics)

        self.metrics = nn.ModuleList(
            [self.get_metric_from_name(metric) for metric in metrics]
        )

    def get_metric_from_name(self, name):
        if name == "PSNR":
            metric = PSNR()
        elif name == "SSIM":
            metric = SSIM()
        else:
            raise "Invalid metric name in QualityMetrics"
        # TODO: If needed, more metric classes can be introduced here later on.

        if torch.cuda.is_available():
            return metric.cuda()

    def forward(self, pred_img, gt_img):
        """
        For each metric function provided, evaluate the function with prediction and target.
        Output is returned in "results" dict.
        
        :param pred_img: NVS image outputted from the generator
            used for loss calculation/metric evaluation
        :param gt_img: GT image for the novel view
            used for loss calculation/metric evaluation
        """

        # Initialize output dict
        results = {}

        for func in self.metrics:
            # Evaluate each different metric (SSIM, PSNR) function between the prediction and target
            out = func(pred_img, gt_img)
            
            # Merge both dicts and store the resulting dict in results
            results.update(out)

        return results # Contains individual metric measurements