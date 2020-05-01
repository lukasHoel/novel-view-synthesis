import torch
import torch.nn as nn

from models.enc_dec.feature_network import FeatureNet
from models.enc_dec.depth_network import Unet
from projection.z_buffer_manipulator import PtsManipulator
from models.enc_dec.refinement_network import RefineNet

class NovelViewSynthesisModel(nn.Module):
    def __init__(self,
                 W,
                 max_z=0,
                 min_z=0,
                 enc_dims=[3, 8, 16, 32],
                 enc_blk_types=["id", "id", "id"],
                 dec_dims=[32, 64, 32, 16, 3],
                 dec_blk_types=["id", "avg", "ups", "id"],
                 use_rgb_features=False,
                 use_gt_depth=False,
                 use_inverse_depth=False,
                 normalize_images=True): # todo W=imageSize for PtsManipulator?
        super().__init__()

        # PARAMETERS
        self.W = W
        self.max_z = max_z
        self.min_z = min_z
        self.enc_dims = enc_dims
        self.enc_blk_types = enc_blk_types
        self.dec_dims = dec_dims
        self.dec_blk_types = dec_blk_types

        # CONTROLS
        self.use_rgb_features = use_rgb_features
        self.use_gt_depth = use_gt_depth
        self.use_inverse_depth = use_inverse_depth
        self.normalize_images = normalize_images

        # ENCODER
        # Encode features to a given resolution
        self.encoder = FeatureNet(res_block_dims=self.enc_dims,
                                  res_block_types=self.enc_blk_types)

        # POINT CLOUD TRANSFORMER
        # REGRESS 3D POINTS
        self.pts_regressor = Unet(num_filters=4, channels_in=3, channels_out=1)

        # TODO is this the class that takes care of ambiguous depth after reprojection?
        '''
        if "modifier" in self.opt.depth_predictor_type:
            self.modifier = Unet(channels_in=64, channels_out=64, opt=opt)
        '''

        # 3D Points transformer
        if self.use_rgb_features:
            self.pts_transformer = PtsManipulator(opt.W, C=3, opt=opt) #todo opt
        else:
            self.pts_transformer = PtsManipulator(opt.W, C=feature_channels, opt=opt) #todo opt

        # DECODER
        # REFINEMENT NETWORK
        self.projector = RefineNet(res_block_dims=self.dec_dims,
                                   res_block_types=self.dec_blk_types)

        # TODO WHERE IS THIS NEEDED?
        '''
        self.min_tensor = self.register_buffer("min_z", torch.Tensor([0.1]))
        self.max_tensor = self.register_buffer(
            "max_z", torch.Tensor([self.opt.max_z])
        )
        self.discretized = self.register_buffer(
            "discretized_zs",
            torch.linspace(self.opt.min_z, self.opt.max_z, self.opt.voxel_size),
        )
        '''

    def forward(self,
                input_img,
                K,
                K_inv,
                input_RT,
                input_RT_inv,
                output_RT,
                output_RT_inv,
                gt_img=None,
                depth_img=None):

        # ENCODE IMAGE
        if self.use_rgb_features:
            img_features = input_img
        else:
            img_features = self.encoder(input_img)

        # GET DEPTH
        if not self.use_gt_depth:
            # predict depth
            regressed_pts = nn.Sigmoid()(self.pts_regressor(input_img))

            # normalize depth
            if not self.use_inverse_depth:
                # Normalize in [min_z, max_z] range
                regressed_pts = regressed_pts * (self.max_z - self.min_z) + self.min_z
            else:
                # Use the inverse for datasets with landscapes, where there
                # is a long tail on the depth distribution
                regressed_pts = 1. / regressed_pts * 10 + 0.01 # todo why these values?
        else:
            if depth_img is None:
                raise ValueError("depth_img must not be None when using gt_depth")
            regressed_pts = depth_img

        # APPLY TRANSFORMATION and REPROJECT
        transformed_img_features = self.pts_transformer.forward_justpts(
            img_features,
            regressed_pts,
            K,
            K_inv,
            input_RT,
            input_RT_inv,
            output_RT,
            output_RT_inv,
        )

        # TODO is this the class that takes care of ambiguous depth after reprojection?
        '''
        if "modifier" in self.opt.depth_predictor_type:
            transformed_img_features = self.modifier(transformed_img_features)
        '''

        # DECODE IMAGE
        transformed_img = self.projector(transformed_img_features)

        # NORMALIZE IMAGES: Output of refinement_network (self.projector) is tanh --> [-1,1] --> normalize to [0,1]
        if self.normalize_images:
            input_img = 0.5 * input_img + 0.5
            gt_img = 0.5 * gt_img + 0.5
            transformed_img = 0.5 * transformed_img + 0.5

        return {
            "InputImg": input_img,
            "OutputImg": gt_img,
            "PredImg": transformed_img,
            "PredDepth": regressed_pts,
        }

    # TODO WHERE IS THIS USED? At inference time for multiple image generations?
    '''
    def forward_angle(self, batch, RTs, return_depth=False):
        # Input values
        input_img = batch["images"][0]

        # Camera parameters
        K = batch["cameras"][0]["K"]
        K_inv = batch["cameras"][0]["Kinv"]

        if torch.cuda.is_available():
            input_img = input_img.cuda()

            K = K.cuda()
            K_inv = K_inv.cuda()

            RTs = [RT.cuda() for RT in RTs]
            identity = (
                torch.eye(4).unsqueeze(0).repeat(input_img.size(0), 1, 1).cuda()
            )

        fs = self.encoder(input_img)
        regressed_pts = (
            nn.Sigmoid()(self.pts_regressor(input_img))
            * (self.opt.max_z - self.opt.min_z)
            + self.opt.min_z
        )

        # Now rotate
        gen_imgs = []
        for RT in RTs:
            torch.manual_seed(
                0
            )  # Reset seed each time so that noise vectors are the same
            gen_fs = self.pts_transformer.forward_justpts(
                fs, regressed_pts, K, K_inv, identity, identity, RT, None
            )

            # now create a new image
            gen_img = self.projector(gen_fs)

            gen_imgs += [gen_img]

        if return_depth:
            return gen_imgs, regressed_pts

        return gen_imgs
    '''
