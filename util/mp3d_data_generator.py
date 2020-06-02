# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image # Added

import numpy as np # Added
from util.mp3d_data_gen_deps.synsin_options import get_dataset
from util.mp3d_data_gen_deps.synsin_train_options import ArgumentParser, get_timestamp
import os


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

def save_data(batch, folder, epoch): # TODO: extend it to n views, n is default 2, focusing on image pairs for now.
    
    img_batch0, img_batch1 = batch["images"]              # List of size n (different views)
    depth_batch0, depth_batch1 = batch["depths"]          # List of size n (different views)
    cam_batch0, cam_batch1 = batch["cameras"]             # List of size n (different views)
    semantic_batch0, semantic_batch1 = batch["semantics"] # List of size n (different views)
    
    batch_size = batch["images"][0].shape[0]
    cam_file_temp = "{:<12} = {}';\n"
    suffix = "img_"

    for batch_idx in range(batch_size):
        # Save RGB images
        img0, img1 = img_batch0[batch_idx].cpu(), img_batch1[batch_idx].cpu()
        save_image(img0, folder + suffix + str(epoch) + "_" + str(batch_idx) + '_0' + '.png')
        save_image(img1, folder + suffix + str(epoch) + "_" + str(batch_idx) + '_1' + '.png')

        # Save depth information in ICL-NUIM conventions        
        depth0, depth1 = depth_batch0[batch_idx].squeeze(0).cpu().numpy().ravel(),\
                         depth_batch1[batch_idx].squeeze(0).cpu().numpy().ravel()
        np.savetxt(folder + suffix + str(epoch) + "_" + str(batch_idx) + '_0' + '.depth', depth0, fmt='%.5f', delimiter=' ', newline=' ')
        np.savetxt(folder + suffix + str(epoch) + "_" + str(batch_idx) + '_1' + '.depth', depth1, fmt='%.5f', delimiter=' ', newline=' ')

        # Save camera parameters in ICL-NUIM conventions
        # NOTE: According to SynSin implementation of get_camera_matrices (@camera_transformations.py):
        # P: World->Cam, Pinv: Cam->World
        # Converting it to ICL conventions
        P0, K0, Pinv0, Kinv0 = cam_batch0["P"][batch_idx],\
                               cam_batch0["K"][batch_idx],\
                               cam_batch0["Pinv"][batch_idx],\
                               cam_batch0["Kinv"][batch_idx]

        cam_pos, cam_up, cam_dir = split_RT(Pinv0.cpu().numpy())
        info = cam_file_temp.format("cam_pos", cam_pos)
        info += cam_file_temp.format("cam_dir", cam_dir)
        info += cam_file_temp.format("cam_up", cam_up)
        with open(folder + suffix + str(epoch) + "_" + str(batch_idx) + '_0' + '.txt', 'w+') as f:
            f.write(info)

        P1, K1, Pinv1, Kinv1 = cam_batch1["P"][batch_idx],\
                               cam_batch1["K"][batch_idx],\
                               cam_batch1["Pinv"][batch_idx],\
                               cam_batch1["Kinv"][batch_idx]

        cam_pos, cam_up, cam_dir = split_RT(Pinv1.cpu().numpy())
        info = cam_file_temp.format("cam_pos", cam_pos)
        info += cam_file_temp.format("cam_dir", cam_dir)
        info += cam_file_temp.format("cam_up", cam_up)
        with open(folder + suffix + str(epoch) + "_" + str(batch_idx) + '_1' + '.txt', 'w+') as f:
            f.write(info)
        
        # TEMP: Save transformation matrices 
        # np.savetxt(folder + "P" + str(epoch) + "_" + str(batch_idx) + '_0' + '.txt', P0)
        # np.savetxt(folder + "K" + str(epoch) + "_" + str(batch_idx) + '_0' + '.txt', K0)
        # np.savetxt(folder + "Pinv" + str(epoch) + "_" + str(batch_idx) + '_0' + '.txt', Pinv0)
        # np.savetxt(folder + "Kinv" + str(epoch) + "_" + str(batch_idx) + '_0' + '.txt', Kinv0)

        # np.savetxt(folder + "P" + str(epoch) + "_" + str(batch_idx) + '_1' + '.txt', P1)
        # np.savetxt(folder + "K" + str(epoch) + "_" + str(batch_idx) + '_1' + '.txt', K1)
        # np.savetxt(folder + "Pinv" + str(epoch) + "_" + str(batch_idx) + '_1' + '.txt', Pinv1)
        # np.savetxt(folder + "Kinv" + str(epoch) + "_" + str(batch_idx) + '_1' + '.txt', Kinv1)


        # Save semantics in the form of int IDs
        semantic0, semantic1 = semantic_batch0[batch_idx].squeeze(0).cpu().numpy().ravel(),\
                               semantic_batch1[batch_idx].squeeze(0).cpu().numpy().ravel()
        np.savetxt(folder + suffix + str(epoch) + "_" + str(batch_idx) + '_0' + '.semantic', semantic0, fmt='%d', delimiter=' ', newline=' ')
        np.savetxt(folder + suffix + str(epoch) + "_" + str(batch_idx) + '_1' + '.semantic', semantic1, fmt='%d', delimiter=' ', newline=' ')


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
      # python data_generator.py --max_epoch 2 --batch-size 5  --render_ids 0 --normalize_image --use_semantics --config full_path_to/pointnav_rgbd.yaml
        # --max_epoch 2 (default: 500)
        # --batch-size 5 (default: 16)
        # --render_ids [0] (default: [0, 1])
        # --normalize_image True (default: False)
        # --use_semantics True (default: False)
        # --config full_path_to/pointnav_rgbd.yaml
      # Flags that can stay the same:
        # --num_workers 0 (default: 0)
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
        if len(os.listdir(dataset_path)) > 0:
            print(dataset_path, "exists and not empty. Aborted.")
            exit(-1)

    for epoch in range(opts.max_epoch):
        batch = next(iter(train_data_loader))
        save_data(batch, dataset_path, epoch)