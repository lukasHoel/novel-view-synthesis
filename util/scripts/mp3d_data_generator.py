# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import multiprocessing
import numpy as np
from util.scripts.mp3d_data_gen_deps.synsin_options import get_dataset
from util.scripts.mp3d_data_gen_deps.synsin_train_options import ArgumentParser
import os
import re


torch.backends.cudnn.benchmark = True

def show(tensor):
    import matplotlib.pyplot as plt
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    
    tensor = tensor.cpu().numpy().transpose(1,2,0)
    plt.imshow(tensor)
    plt.show()

def split_RT(RT):
    formatter={'float_kind':lambda x: "%.10f" % x}
    R = RT[0:3, 0:3]
    cam_pos = RT[0:3, 3].ravel()
    cam_up = R[:, 1].ravel()  # y=cam_up (already unit)
    cam_dir = R[:, 2].ravel() # z=cam_dir (already unit)
    cam_pos = np.array2string(cam_pos, formatter=formatter, max_line_width=np.inf, separator=", ")
    cam_up = np.array2string(cam_up, formatter=formatter, max_line_width=np.inf, separator=", ")
    cam_dir = np.array2string(cam_dir, formatter=formatter, max_line_width=np.inf, separator=", ")
    return cam_pos, cam_up, cam_dir

def save_data(batch,
              folder,
              scene_id,
              batch_idx,
              use_txt_depth=True,
              use_binary_depth=True, 
              use_semantics=True):

    # Create subfolder using ID of the scene
    full_path = os.path.join(folder, scene_id)
    print("\nSaving under directory:", full_path)
    
    try:
        os.mkdir(full_path)
    except FileExistsError:
        pass

    # Each value in batch dict: List of size n (different views)
    # NOTE: n is default 2, focusing on image pairs for now.
    img_batch0, img_batch1 = batch["images"]
    cam_batch0, cam_batch1 = batch["cameras"]
    depth_batch0, depth_batch1 = batch["depths"] if use_txt_depth or use_binary_depth else (None, None)
    semantic_batch0, semantic_batch1 = batch["semantics"] if use_semantics else (None, None)
    
    file_prefix = os.path.join(full_path, scene_id)
    cam_file_content = "{:<12} = {}';\n"
    sample_batch_size = batch["images"][0].shape[0]

    start_idx = batch_idx * sample_batch_size
    num_views = 2
    exts = "png,txt"
    files_per_view = 2

    if use_semantics:
        exts+=",semantic"
        files_per_view += 1

    if use_txt_depth:
        exts+=",depth"
        files_per_view += 1

    if use_binary_depth:
        exts+=",depth.npy"
        files_per_view += 1

    exts = "{" + exts + "}"
    files_per_sample = num_views * files_per_view

    for sample_idx in range(sample_batch_size):
        curr_file_idx = str(start_idx + sample_idx)
        template = file_prefix + "_" + curr_file_idx + "_{pair_id}.{ext}"
        
        # Save RGB images (scene_idx_pairid.png)
        img0, img1 = img_batch0[sample_idx].cpu(), img_batch1[sample_idx].cpu()
        save_image(img0, template.format(pair_id=0, ext='png'))
        save_image(img1, template.format(pair_id=1, ext='png'))

        if use_txt_depth or use_binary_depth:
            depth0, depth1 = depth_batch0[sample_idx].squeeze(0).cpu().numpy(),\
                             depth_batch1[sample_idx].squeeze(0).cpu().numpy()

            # Save depth information as text (scene_idx_pairid.depth)
            if use_txt_depth:
                np.savetxt(template.format(pair_id=0, ext='depth'), depth0.ravel(), fmt='%.5f', delimiter=' ', newline=' ')
                np.savetxt(template.format(pair_id=1, ext='depth'), depth1.ravel(), fmt='%.5f', delimiter=' ', newline=' ')
                
            # Save depth information as binary file (scene_idx_pairid.depth.npy)
            if use_binary_depth:
                np.save(template.format(pair_id=0, ext='depth.npy'), depth0)
                np.save(template.format(pair_id=1, ext='depth.npy'), depth1)

        # Save camera parameters (scene_idx_pairid.txt)
        # NOTE: According to SynSin implementation of get_camera_matrices (@camera_transformations.py):
        # P: World->Cam, Pinv: Cam->World
        P0, K0, Pinv0, Kinv0 = cam_batch0["P"][sample_idx],\
                               cam_batch0["K"][sample_idx],\
                               cam_batch0["Pinv"][sample_idx],\
                               cam_batch0["Kinv"][sample_idx]

        cam_pos, cam_up, cam_dir = split_RT(Pinv0.cpu().numpy())
        info = cam_file_content.format("cam_pos", cam_pos)
        info += cam_file_content.format("cam_dir", cam_dir)
        info += cam_file_content.format("cam_up", cam_up)
        with open(template.format(pair_id=0, ext='txt'), 'w+') as f:
            f.write(info)

        P1, K1, Pinv1, Kinv1 = cam_batch1["P"][sample_idx],\
                               cam_batch1["K"][sample_idx],\
                               cam_batch1["Pinv"][sample_idx],\
                               cam_batch1["Kinv"][sample_idx]

        cam_pos, cam_up, cam_dir = split_RT(Pinv1.cpu().numpy())
        info = cam_file_content.format("cam_pos", cam_pos)
        info += cam_file_content.format("cam_dir", cam_dir)
        info += cam_file_content.format("cam_up", cam_up)
        with open(template.format(pair_id=1, ext='txt'), 'w+') as f:
            f.write(info)

        # Save semantics in the form of int IDs (scene_idx_pairid.semantic)
        if use_semantics:
            semantic0, semantic1 = semantic_batch0[sample_idx].squeeze(0).cpu().numpy().ravel(),\
                                   semantic_batch1[sample_idx].squeeze(0).cpu().numpy().ravel()
            np.savetxt(template.format(pair_id=0, ext='semantic'), semantic0, fmt='%d', delimiter=' ', newline=' ')
            np.savetxt(template.format(pair_id=1, ext='semantic'), semantic1, fmt='%d', delimiter=' ', newline=' ')

        print("Files created: {}_{}_{{0,1}}.{}".format(scene_id, curr_file_idx, exts))

    print("Saving completed. Number of files created under {}: {}\n".format(full_path, files_per_sample * sample_batch_size))


if __name__ == "__main__":

    ####################
    # Populate dataset #
    ####################
    # Prerequisites before running the script
    # Install frameworks: 
    # (1) habitat-sim: https://github.com/facebookresearch/habitat-sim#installation 
    # (2) habitat-api: https://github.com/facebookresearch/habitat-api#installation
    # Install data: 
    # - MatterPort3D: https://github.com/facebookresearch/habitat-api#data
    # - Point goal navigation: https://github.com/facebookresearch/habitat-api#task-datasets
    # conda activate habitat

    # Relevant flags:
    # Modifiable @train_options.py:
      # Flags to modify:
      # python mp3d_data_generator.py --max_runs 2 --sample_batch_count 5 --sample_batch_size 5 --normalize_image --data_render_path X --habitat_api_prefix Y
        # --max_runs 2 (default: 90)
        # --sample_batch_count 5 (default: 1)
        # --sample_batch_size 5 (default: 100)
        # --normalize_image True (default: False)
        # --envs scene_id1 scene_id2 (default: []) 
        # --data_render_path path_to_dataset_to_be_generated (required, e.g. "./data/mp3d_dataset")
        # --habitat_api_prefix path_to_habitat-api (required, e.g. "'path_prefix' part of path_prefix/habitat-api")
      # Flags that can stay the same:
        # --no_txt_depth (default: False)
        # --no_binary_depth (default: False)
        # --no_semantics (default: False)
        # --num_workers 1 (default: 1)
        # --render_ids 0 (default: [0])
        # --gpu_ids 0 (default: 0)
        # --dataset 'mp3d' (default: 'mp3d')
        # --image_type "both" (default: "both")
        # --num_views 2 (default: 2)
        # --images_before_reset 1000 (default: 1000)
        # --seed 0 (default: 0)
        # -W 256 (default:256)

    opts, _ = ArgumentParser().parse()

    manager = multiprocessing.Manager()
    envs_processed = manager.list()
    envs_to_process = manager.list(opts.envs)

    Dataset = get_dataset(opts)
    train_set = Dataset("train", envs_processed, envs_to_process, opts)

    train_data_loader = DataLoader(
        dataset=train_set,
        num_workers=opts.num_workers,
        batch_size=opts.sample_batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=False,
    )
    dataset_path = opts.data_render_path
    try:
        os.mkdir(dataset_path)
    except FileExistsError:
        pass

    use_txt_depth = not opts.no_txt_depth
    use_binary_depth = not opts.no_binary_depth
    use_semantics = not opts.no_semantics

    try:
        for scene_idx in range(opts.max_runs):
            batch_iter = iter(train_data_loader)

            for batch_idx in range(opts.sample_batch_count):
                batch = next(batch_iter)
                scene_id = batch["scene_path"][0].split("/")[-2]
                save_data(batch, dataset_path, scene_id, batch_idx, use_txt_depth, use_binary_depth, use_semantics)

                # Keep track of processed envs
                envs_processed.append(scene_id)

    except AssertionError:
        print("No more envs to process")