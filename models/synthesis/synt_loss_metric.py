from models.synthesis.losses import *
from models.synthesis.metrics import *

class SceneEditingAndSynthesisLoss(nn.Module):
    """
    Class for simultaneous calculation of SynthesisLoss and SceneEditingLoss.
    """
    def __init__(self,
                 synthesis_losses=['1.0_l1', '10.0_content'],
                 synthesis_ignore_at_scene_editing_masks = True,
                 scene_editing_lambdas=[1.0, 1.0],
                 scene_editing_lpregion_params=[1.0, 0.0, 2.0]):
        super().__init__()

        self.synthesis_loss = SynthesisLoss(synthesis_losses, synthesis_ignore_at_scene_editing_masks)
        self.scene_editing_loss = SceneEditingLoss(scene_editing_lambdas, scene_editing_lpregion_params)

    def forward(self, pred_img, gt_img, input_mask, gt_output_mask):
        synthesis_results = self.synthesis_loss(pred_img, gt_img, input_mask, gt_output_mask)
        scene_editing_results = self.scene_editing_loss(pred_img, gt_img, input_mask, gt_output_mask)

        results = {**synthesis_results, **scene_editing_results}
        results["Total Loss"] = synthesis_results["Total Loss"] + scene_editing_results["Total Loss"]

        return results

class SceneEditingLoss(nn.Module):
    """
    Calculates the RegionSimilarityLoss and LPRegionLoss over the segmentation prediction at the given movement masks.
    """

    def __init__(self, lambdas=[1.0, 1.0], lpregion_params=[1.0, 0.0, 2.0]):
        super().__init__()

        self.lambdas = lambdas
        self.region_similarity = RegionSimilarityLoss()
        self.region_lp = LPRegionLoss(*lpregion_params)

        if torch.cuda.is_available():
            self.region_similarity = self.region_similarity.cuda()
            self.region_lp = self.region_lp.cuda()

    def calculate_predicted_mask(self, pred_img, gt_img, gt_output_mask):
        # TODO how to vectorize this?
        bs = gt_output_mask.shape[0]
        pred_output_mask = torch.zeros_like(gt_output_mask)
        for i in range(bs):
            # get the first position where mask is true (nonzero)
            # TODO support more than one color --> find all colors in gt_output_mask @ gt_img
            # TODO what if in gt image the color is not consistent as well e.g. due to downsampling + antialiasing???
            color_index = torch.nonzero(gt_output_mask[i].squeeze(), as_tuple=False)[30]

            # get the color from gt_img at that position
            color = gt_img[i, :, color_index[0], color_index[1]].unsqueeze(1).unsqueeze(2)

            # find all places where the color is equal in pred_img (comparison is per channel here)
            # TODO add "nearest color" search if color is not exactly the same? Does this make sense. We could see it as punishment when color is not exactly similar?
            # But we could use gradients when we search for exact same color and it is not exactly the same
            # Also: Floating point precision???
            color_channel_equal = torch.eq(gt_img[i], color)

            # merge to final mask where all 3 color channels are indeed equal
            color_equal = color_channel_equal[0] & color_channel_equal[1] & color_channel_equal[2]

            # set this as mask for the i-th batch
            pred_output_mask[i] = color_equal

            '''
            import matplotlib.pyplot as plt
            import numpy as np
            plt.imshow(pred_img[i].permute((1,2,0)).cpu().detach().numpy())
            plt.show()

            plt.imshow(gt_img[i].permute((1, 2, 0)).cpu().detach().numpy())
            plt.show()

            plt.imshow(np.moveaxis(gt_output_mask[i].cpu().detach().numpy(), 0, -1).squeeze())
            plt.show()

            plt.imshow(np.moveaxis(pred_output_mask[i].cpu().detach().numpy(), 0, -1).squeeze())
            plt.show()
            '''

        return pred_output_mask

    def forward(self, pred_img, gt_img, input_mask, gt_output_mask):
        # calculate predicted_mask
        pred_output_mask = self.calculate_predicted_mask(pred_img, gt_img, gt_output_mask)

        # pass to region similarity loss
        region_sim = self.region_similarity(pred_output_mask, gt_output_mask)

        # pass to lp region loss: lower_region is input mask and higher_region is merged_output_mask
        # (this is the convention of this loss) TODO makes this sense?
        # Alternative: higher region is merged_output_mask + input_mask, lower_region is rest of the image
        merged_output_mask = (pred_output_mask == True) | (gt_output_mask == True)
        region_lp = self.region_lp(pred_img, gt_img, input_mask, merged_output_mask)

        # create dict containing both results
        result = {**region_sim, **region_lp}
        result["Total Loss"] = region_sim["Total Loss"] * self.lambdas[0] + region_lp["Total Loss"] * self.lambdas[1]

        return result

class SynthesisLoss(nn.Module):
    """
    Class for simultaneous calculation of L1, content/perceptual losses.
    Losses to use should be passed as argument.
    """
    def __init__(self, losses=['1.0_l1', '10.0_content'], ignore_at_scene_editing_masks=False):
        """
        :param losses: 
            loss specification, str of the form: 'lambda_loss'
            lambda is used to weight different losses
            l1 and content/perceptual losses are summed if both are specified
            used in the forward method

        :param ignore_at_scene_editing_masks:
            If true, we do not calculate the loss for the input and output mask of scene editing movements.
            We do this by setting the pred_img equal to the gt_img at these mask positions, s.t. the loss will be 0 at these locations.
            If false (default), we calculate the losses over the whole image as it is and we ignore the masks.
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
        self.ignore_at_scene_editing_masks = ignore_at_scene_editing_masks

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

    def forward(self, pred_img, gt_img, input_mask=None, gt_output_mask=None):
        """
        For each loss function provided, evaluate the function with prediction and target.
        Results of individual functions along with the total loss returned in a dictionary.
        
        :param pred_img: NVS image outputted from the generator
            used for loss calculation/metric evaluation
        :param gt_img: GT image for the novel view
            used for loss calculation/metric evaluation
        :param input_mask: from which pixels were objects moved to another location
        :param gt_output_mask: to which pixels should objects be moved to after applying movements
        """

        if self.ignore_at_scene_editing_masks:
            bs = pred_img.shape[0]
            # TODO how to vectorize this?
            pred_img_masked = pred_img.clone() # fix for backpropagation: inplace operations like in next line(s) do not work with pytorch autograd
            for i in range(bs):
                in_mask = input_mask[i].squeeze()
                out_mask = gt_output_mask[i].squeeze()
                pred_img_masked[i, :, in_mask] = gt_img[i, :, in_mask]
                pred_img_masked[i, :, out_mask] = gt_img[i, :, out_mask]

                '''
                import matplotlib.pyplot as plt
                plt.imshow(pred_img_masked[i].permute((1, 2, 0)).cpu().detach().numpy())
                plt.show()

                plt.imshow(pred_img[i].permute((1, 2, 0)).cpu().detach().numpy())
                plt.show()

                print("mask in")
                plt.imshow(in_mask.cpu().detach().numpy())
                plt.show()

                print("mask out")
                plt.imshow(out_mask.cpu().detach().numpy())
                plt.show()
                '''

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