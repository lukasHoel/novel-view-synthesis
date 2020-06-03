# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image # Added

import numpy as np # Added
from util.mp3d_data_gen_deps.synsin_options import get_dataset
from util.mp3d_data_gen_deps.synsin_train_options import ArgumentParser, get_timestamp
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

def save_data(batch, folder): # TODO: extend it to n views, n is default 2, focusing on image pairs for now.
    # Create subfolder using name of the scene
    regex = r".*/(\w+)\.glb"
    scene_path = batch["scene_path"][0]           # List of size batch_size
    scene = re.search(regex, scene_path).group(1)
    full_path = os.path.join(folder, scene)
    file_idx = 0
    num_files = 0

    print("\nSaving under directory:", full_path)
    
    try:
        os.mkdir(full_path)
    except FileExistsError:
        files = os.listdir(full_path)
        num_files = len(files)
        # Check if there exist saved files from that scene, last saved index, starting indexing from there
        if num_files > 0:
            regex = r"img_(\d+)_\d+\.\w+"                            # Regex to extract data indices
            func = lambda text: int(re.search(regex, text).group(1)) # Extract each index
            file_idx = max(map(func, files)) + 1                     # Find last used index, use +1
            print("Warning: {} exists:\n* Number of files: {}\n* Last used index under path: {}".format(full_path, num_files, file_idx-1))

    img_batch0, img_batch1 = batch["images"]              # List of size n (different views)
    depth_batch0, depth_batch1 = batch["depths"]          # List of size n (different views)
    cam_batch0, cam_batch1 = batch["cameras"]             # List of size n (different views)
    semantic_batch0, semantic_batch1 = batch["semantics"] # List of size n (different views)
    
    file_prefix = os.path.join(full_path, "img_")
    cam_file_content = "{:<12} = {}';\n"
    batch_size = batch["images"][0].shape[0]

    for batch_idx in range(batch_size):
        curr_file_idx = str(file_idx + batch_idx)
        template = file_prefix + curr_file_idx + "_{pair_id}.{ext}"
        
        # Save RGB images (img_idx_pairid.png)
        img0, img1 = img_batch0[batch_idx].cpu(), img_batch1[batch_idx].cpu()
        save_image(img0, template.format(pair_id=0, ext='png'))
        save_image(img1, template.format(pair_id=1, ext='png'))

        # Save depth information (img_idx_pairid.depth)
        depth0, depth1 = depth_batch0[batch_idx].squeeze(0).cpu().numpy().ravel(),\
                         depth_batch1[batch_idx].squeeze(0).cpu().numpy().ravel()
        np.savetxt(template.format(pair_id=0, ext='depth'), depth0, fmt='%.5f', delimiter=' ', newline=' ')
        np.savetxt(template.format(pair_id=1, ext='depth'), depth1, fmt='%.5f', delimiter=' ', newline=' ')

        # Save camera parameters (img_idx_pairid.txt)
        # NOTE: According to SynSin implementation of get_camera_matrices (@camera_transformations.py):
        # P: World->Cam, Pinv: Cam->World
        P0, K0, Pinv0, Kinv0 = cam_batch0["P"][batch_idx],\
                               cam_batch0["K"][batch_idx],\
                               cam_batch0["Pinv"][batch_idx],\
                               cam_batch0["Kinv"][batch_idx]

        cam_pos, cam_up, cam_dir = split_RT(Pinv0.cpu().numpy())
        info = cam_file_content.format("cam_pos", cam_pos)
        info += cam_file_content.format("cam_dir", cam_dir)
        info += cam_file_content.format("cam_up", cam_up)
        with open(template.format(pair_id=0, ext='txt'), 'w+') as f:
            f.write(info)

        P1, K1, Pinv1, Kinv1 = cam_batch1["P"][batch_idx],\
                               cam_batch1["K"][batch_idx],\
                               cam_batch1["Pinv"][batch_idx],\
                               cam_batch1["Kinv"][batch_idx]

        cam_pos, cam_up, cam_dir = split_RT(Pinv1.cpu().numpy())
        info = cam_file_content.format("cam_pos", cam_pos)
        info += cam_file_content.format("cam_dir", cam_dir)
        info += cam_file_content.format("cam_up", cam_up)
        with open(template.format(pair_id=1, ext='txt'), 'w+') as f:
            f.write(info)

        # Save semantics in the form of int IDs (img_idx_pairid.semantic)
        semantic0, semantic1 = semantic_batch0[batch_idx].squeeze(0).cpu().numpy().ravel(),\
                               semantic_batch1[batch_idx].squeeze(0).cpu().numpy().ravel()
        np.savetxt(template.format(pair_id=0, ext='semantic'), semantic0, fmt='%d', delimiter=' ', newline=' ')
        np.savetxt(template.format(pair_id=1, ext='semantic'), semantic1, fmt='%d', delimiter=' ', newline=' ')

        print("Files created: img_{}_{{0,1}}.{{png,depth,txt,semantic}}".format(curr_file_idx))
        num_files += 8

    print("Saving completed. Total number of files under {}: {}\n".format(full_path, num_files))


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
      # python data_generator.py --max_epoch 2 --batch-size 5 --normalize_image --use_semantics --config full_path_to/pointnav_rgbd.yaml
        # --max_epoch 2 (default: 500)
        # --batch-size 5 (default: 16)
        # --normalize_image True (default: False)
        # --use_semantics True (default: False)
        # --config full_path_to/pointnav_rgbd.yaml
      # Flags that can stay the same:
        # --num_workers 1 (default: 1)
        # --render_ids 0 (default: [0])
        # --gpu_ids 0 (default: 0)
        # --dataset 'mp3d' (default: 'mp3d')
        # --image_type "both" (default: "both")
        # --num_views 2 (default: 2)
        # --images_before_reset 1000 (default: 1000)
        # --seed 0 (default: 0)
        # -W 256 (default:256)
    # Static 'flags' to be adjusted @options.py
        # train_data_path full_path_to/train.json.gz
        # val_data_path full_path_to/val.json.gz
        # test_data_path full_path_to/test.json.gz
        # scenes_dir full_path_to/scene_datasets (which is top level directory containing 90 folders each of which include following 4 files *.glb *.house *.navmesh *.ply)

    # Habitat behaviour depending on --num_workers:
        # num_worker: 0 -> Use single env for all epochs: Keep sampling from the same env througout training
        # num_worker: 1 -> Use single env within an epoch: Sample batch from same env, but change env to sample at every epoch.
        # num_worker: 2 -> Run 2 different envs at every epoch epoch, but sample the batch from one of these only (probably from env of the default agent), change envs over epochs

    opts, _ = ArgumentParser().parse()

    timestamp = get_timestamp()
    print("Timestamp ", timestamp, flush=True)

    Dataset = get_dataset(opts)
    train_set = Dataset("train", opts)

    train_data_loader = DataLoader(
        dataset=train_set,
        num_workers=opts.num_workers,
        batch_size=opts.batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=False,
    )
    dataset_path = "./data/mp3d_dataset/"
    try:
        os.mkdir(dataset_path)
    except FileExistsError:
        pass

    for _ in range(opts.max_epoch):
        batch = next(iter(train_data_loader))
        save_data(batch, dataset_path)