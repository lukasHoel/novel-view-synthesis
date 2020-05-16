import torch
import torch.nn as nn

EPS = 1e-2

def get_splatter(
        name, depth_values, opt=None, size=256, C=3, points_per_pixel=8
):
    if name == "xyblending":
        from projection.z_buffer_layers import RasterizePointsXYsBlending

        return RasterizePointsXYsBlending(
            C=C,
            # learn_feature=opt.learn_default_feature,
            # radius=opt.radius,
            size=size,
            points_per_pixel=points_per_pixel,
            # opts=opt,
            accumulation_tau=1
        )
    # TODO: think about adding new parameters from the adapted version of this class (due to removal of opt...)
    # New parameters are: rad_pow, accumulation, accumulation_tau (see also paper equations 1 and 2)

    else:
        raise NotImplementedError()


class PtsManipulator(nn.Module):
    def __init__(self, W, H, C=3):
        super().__init__()

        self.splatter = get_splatter(
            "xyblending", None, size=W, C=C, points_per_pixel=8
        )

        self.img_shape = (H, W)

        # create coordinate system for x and y
        xs = torch.linspace(0, W - 1, W)
        ys = torch.linspace(0, H - 1, H)

        xs = xs.view(1, 1, 1, W).repeat(1, 1, H, 1)
        ys = ys.view(1, 1, H, 1).repeat(1, 1, 1, W)

        # build homogeneous coordinate system with [X, Y, 1, 1] to prepare for depth
        xyzs = torch.cat(
            (xs, ys, torch.ones(xs.size()), torch.ones(xs.size())), 1
        ).view(1, 4, -1)

        self.register_buffer("xyzs", xyzs)

    def project_pts(
            self, pts3D, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2, colors=None
    ):
        # add Zs to the coordinate system
        # projected_coors is then [X*Z, -Y*Z, -Z, 1] with Z being the depth of the image
        projected_coors = self.xyzs * pts3D
        projected_coors[:, -1, :] = 1

        # Transform into camera coordinate of the first view
        cam1_X = K_inv.bmm(projected_coors)

        # Transform to World Coordinates with RT of input view
        wrld_X = RT_cam1.bmm(cam1_X)

        # Transform from World coordinates to camera of output view
        new_coors = RTinv_cam2.bmm(wrld_X)

        # Apply intrinsics / go back to image plane
        xy_proj = K.bmm(new_coors)

        # remove invalid zs that cause nans
        mask = (xy_proj[:, 2:3, :].abs() < EPS).detach()
        zs = xy_proj[:, 2:3, :]
        zs[mask] = EPS

        # here we concatenate (x,y) / -z and the original z-coordinate into a new (x,y,z) vector
        sampler = torch.cat((xy_proj[:, 0:2, :] / zs, xy_proj[:, 2:3, :]), 1)

        # rescale coordinates to work with splatting and move to origin
        sampler[:, 0, :] = sampler[:, 0, :] / float(self.img_shape[1] - 1) * 2 - 1
        sampler[:, 1, :] = sampler[:, 1, :] / float(self.img_shape[0] - 1) * 2 - 1

        # here we set (x,y,z) to -10 where we have invalid zs that cause nans
        sampler[mask.repeat(1, 3, 1)] = -10

        # Don't flip the ys
        # sampler = sampler * torch.Tensor([1, 1, 1]).unsqueeze(0).unsqueeze(2).to(sampler.device)

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

        pts3D = self.project_pts(
            pred_pts, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2
        )
        pointcloud = pts3D.permute(0, 2, 1).contiguous()
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