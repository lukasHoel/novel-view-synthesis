from models.synthesis.losses import *
from models.synthesis.metrics import *

class SynthesisRegionLoss(nn.Module):
    """
    Class for simultaneous calculation of normal SynthesisLoss (L1, content/perceptual) and region-based losses (LPRegionLoss).
    Losses to use should be passed as argument.
    """
    def __init__(self,
                 non_region_losses=['1.0_l1', '10.0_content'],
                 region_losses={"1.0_lp": [1.0, 0.0, 2.0]}):
        """

        Supported region_losses:
            - lp: LPRegionLoss

        :param non_region_losses: used to create a SynthesisLoss. See that class for explanation of argument.
        :param region_losses: dict where each key is a <weight>_<name> pair as for the SynthesisLoss.
                Each value is an implementation-specific list that can be passed as-is to the specific class, i.e.
                we can construct LPRegionLoss with: LPRegionLoss(*value) where value is taken from the dict with a
                corresponding key.
        """
        super().__init__()

        self.normal = SynthesisLoss(non_region_losses)
        lambdas, loss_names = zip(
            *[loss_name.split("_") for loss_name in region_losses.keys()])  # Parse lambda and loss_names from dict keys
        print("Loss names:", loss_names)
        print("Weight of each loss:", lambdas)
        lambdas = [float(l) for l in lambdas]  # [str] -> [float]

        self.lambdas = lambdas
        self.losses = nn.ModuleList(
            [self.get_loss_from_dict(k,v) for k,v in region_losses]
        )

    def get_loss_from_dict(self, key, value):
        if key == "lp":
            loss = LPRegionLoss(*value)
        else:
            raise ValueError("Invalid loss name in SynthesisRegionLoss: " + key)
        # TODO: If needed, more loss classes can be introduced here later on.

        if torch.cuda.is_available():
            return loss.cuda()
        else:
            return loss

    def forward(self, pred_img, gt_img, mask_lower_region, mask_higher_region):
        results = self.normal(pred_img, gt_img)

        for i, func in enumerate(self.losses):
            # Evaluate each different loss function with the prediction and target and masks
            out = func(pred_img, gt_img, mask_lower_region, mask_higher_region)

            # Add the contribution by each loss to the total loss wrt their weights (lambda)
            results["Total Loss"] += out["Total Loss"] * self.lambdas[i]

            # Merge both dicts and store the resulting dict in results
            results = dict(out, **results)

        return results  # Contains individual losses and weighted sum of these

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
            raise ValueError("Invalid loss name in SynthesisLoss: " + name)
        # TODO: If needed, more loss classes can be introduced here later on.

        if torch.cuda.is_available():
            return loss.cuda()
        else:
            return loss

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
            raise ValueError("Invalid metric name in QualityMetrics: " + name)
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