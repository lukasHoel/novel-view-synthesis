# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn

from pytorch3d.structures import Pointclouds

EPS = 1e-2


def get_splatter(
    name, depth_values, opt=None, size=256, C=64, points_per_pixel=8
):
    if name == "xyblending":
        from projection.z_buffer_layers import RasterizePointsXYsBlending

        return RasterizePointsXYsBlending(
            C=C,
            #learn_feature=opt.learn_default_feature,
            #radius=opt.radius,
            size=size,
            points_per_pixel=points_per_pixel,
            #opts=opt,
        )
    # TODO: think about adding new parameters from the adapted version of this class (due to removal of opt...)
    # New parameters are: rad_pow, accumulation, accumulation_tau (see also paper equations 1 and 2)

    else:
        raise NotImplementedError()


class PtsManipulator(nn.Module):
    def __init__(self, W, C=64, opt=None):
        super().__init__()
        self.opt = opt

        self.splatter = get_splatter(
            opt.splatter, None, opt, size=W, C=C, points_per_pixel=opt.pp_pixel
        )

        xs = torch.linspace(0, W - 1, W) / float(W - 1) * 2 - 1
        ys = torch.linspace(0, W - 1, W) / float(W - 1) * 2 - 1

        xs = xs.view(1, 1, 1, W).repeat(1, 1, W, 1)
        ys = ys.view(1, 1, W, 1).repeat(1, 1, 1, W)

        xyzs = torch.cat(
            (xs, -ys, -torch.ones(xs.size()), torch.ones(xs.size())), 1
        ).view(1, 4, -1)

        # TODO: What is xyzs?

        self.register_buffer("xyzs", xyzs)

    def project_pts(
        self, pts3D, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2
    ):
        # PERFORM PROJECTION
        # Project the world points into the new view
        projected_coors = self.xyzs * pts3D # TODO what is the output of this?
        projected_coors[:, -1, :] = 1

        # Transform into camera coordinate of the first view
        cam1_X = K_inv.bmm(projected_coors)

        # Transform into world coordinates
        RT = RT_cam2.bmm(RTinv_cam1)

        wrld_X = RT.bmm(cam1_X)

        # And intrinsics
        xy_proj = K.bmm(wrld_X)

        # And finally we project to get the final result
        mask = (xy_proj[:, 2:3, :].abs() < EPS).detach()

        # Remove invalid zs that cause nans
        zs = xy_proj[:, 2:3, :]
        zs[mask] = EPS

        sampler = torch.cat((xy_proj[:, 0:2, :] / -zs, xy_proj[:, 2:3, :]), 1) # TODO what is happening here?
        sampler[mask.repeat(1, 3, 1)] = -10
        # Flip the ys
        sampler = sampler * torch.Tensor([1, -1, -1]).unsqueeze(0).unsqueeze(
            2
        ).to(sampler.device)

        return sampler

    def forward_justpts(
        self, src, pred_pts, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2
    ):
        # Now project these points into a new view
        bs, c, w, h = src.size()

        if len(pred_pts.size()) > 3:
            # reshape into the right positioning
            pred_pts = pred_pts.view(bs, 1, -1)
            src = src.view(bs, c, -1)

        # pred_pts.shape = (bs, 1, w*h) --> one depth value for every element in the image raster (w*h)
        # src.shape = (bs, c, w*h) --> c features for every element in the image raster (w*h)

        pts3D = self.project_pts(
            pred_pts, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2
        )
        pointcloud = pts3D.permute(0, 2, 1).contiguous()

        # pts3D.shape = (bs, w*h, 3) --> (x,y,z) coordinate for ever element in the image raster
        # Because we have done re-projection, the i-th coordinate in the image raster must no longer be identical to (x,y)!

        result = self.splatter(pointcloud, src)

        return result

    def forward(
        self,
        alphas,
        src,
        pred_pts,
        K,
        K_inv,
        RT_cam1,
        RTinv_cam1,
        RT_cam2,
        RTinv_cam2,
    ):
        # Now project these points into a new view
        bs, c, w, h = src.size()

        if len(pred_pts.size()) > 3:
            # reshape into the right positioning
            pred_pts = pred_pts.view(bs, 1, -1)
            src = src.view(bs, c, -1)
            alphas = alphas.view(bs, 1, -1).permute(0, 2, 1).contiguous()

        pts3D = self.project_pts(
            pred_pts, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2
        )
        result = self.splatter(pts3D.permute(0, 2, 1).contiguous(), alphas, src)

        return result
