import numpy as np
import os
import torch
from torch.utils.data import Dataset
from skimage import io
from util.camera_transformations import *
import re


class ICLNUIMDataset(Dataset):
    '''
    Loads samples from the pre-rendered NUIM dataset: https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html
    Adapted from c++ code in https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html
    '''

    # intrinsic camera matrix taken from https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html
    cam_K = {
        'fx': 481.2,
        'fy': -480.0,
        'cx': 319.5,
        'cy': 239.5
    }

    # intrinsic camera matrix as numpy array
    K = np.zeros((3,3))
    K[0,0] = cam_K['fx']
    K[1,1] = cam_K['fy']
    K[0,2] = cam_K['cx']
    K[1,2] = cam_K['cy']
    K[2,2] = 1

    # regex to read lines from the camera .txt file
    cam_pattern = "(?P<id>.*\w).*= \[(?P<x>.*), (?P<y>.*), (?P<z>.*)\].*"

    def __init__(self, path, depth_to_image_plane=True, sampleOutput=True, transform=None, cam_transforms=False):
        '''

        :param path: path/to/NUIM/files. Needs to be a directory with .png, .depth and .txt files, as can be obtained from: https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html
        :param depth_to_image_plane: whether or not to convert to depth in the .depth file into image_plane depth, see: https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html
        :param sampleOutput: whether or not to uniformly sample a second image + extrinsic camera pose (R|T) in the neighborhood of each accessed item.
                neighborhood is currently defined as: select uniformly at random any camera in the range [index-30, index+30) where index is the accessed item index.
                For example: If the 500. item is accessed, the second camera pose (R|T) will be from any of the poses of the items 470-530 (excluding 500).
        :param transform: transform that should be applied to the input image AND the target depth
        '''
        self.transform = transform
        self.cam_transforms = cam_transforms
        self.depth_to_image_plane = depth_to_image_plane
        self.path = path

        self.img = sorted([f for f in os.listdir(path) if f.endswith('.png')])
        self.depth = sorted([f for f in os.listdir(path) if f.endswith('.depth')])
        self.cam = sorted([f for f in os.listdir(path) if f.endswith('.txt')])

        self.size = len(self.img)

        self.sampleOutput = sampleOutput

    def load_image(self, idx):
        return io.imread(os.path.join(self.path, self.img[idx]))

    def load_depth(self, idx, img_shape=(480, 640)):
        with open(os.path.join(self.path, self.depth[idx])) as f:
            depth = [float(i) for i in f.read().split(' ') if i.strip()]  # read .depth file
            depth = np.asarray(depth, dtype=np.float32).reshape(img_shape)  # convert to same format as image WxH

            # convert to image plane depth by taking into account the position in the WxH array as (x, y)
            if self.depth_to_image_plane:
                depth = np.fromfunction(lambda x, y: self.toImagePlane(depth, x, y), depth.shape, dtype=depth.dtype)
        return depth

    def load_cam(self, idx):
        cam = {} # load the .txt file in this dict
        with open(os.path.join(self.path, self.cam[idx])) as f:
            for line in f:
                m = re.match(ICLNUIMDataset.cam_pattern, line) # will match everything except angle, but that is not needed anyway
                if m is not None:
                    cam[m["id"]] = np.zeros(3)
                    cam[m["id"]][0] = float(m["x"])
                    cam[m["id"]][1] = float(m["y"])
                    cam[m["id"]][2] = float(m["z"])

        # calculate RT matrix, taken from: https://www.doc.ic.ac.uk/~ahanda/VaFRIC/computeRT.m
        z = cam["cam_dir"] / np.linalg.norm(cam["cam_dir"])
        x = np.cross(cam["cam_up"], z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        RT = np.column_stack((x, y, z, cam["cam_pos"]))
        # RT = np.vstack([RT, [0,0,0,1]]) # if (0 0 0 1) row is needed

        # Code to calculate K - unnecessary because K is constant. taken from: https://www.doc.ic.ac.uk/~ahanda/VaFRIC/getcamK.m
        '''
        focal = np.linalg.norm(cam["cam_dir"])
        aspect = np.linalg.norm(cam["cam_right"]) / np.linalg.norm(cam["cam_up"])
        angle = 2 * np.arctan((np.linalg.norm(cam["cam_right"]) / 2) / np.linalg.norm(cam["cam_dir"]))

        M = 480
        N = 640

        width = N
        height = M

        psx = 2 * focal * np.tan(0.5 * angle) / N
        psy = 2 * focal * np.tan(0.5 * angle) / aspect / M

        psx = psx / focal
        psy = psy / focal

        Ox = (width + 1) * 0.5
        Oy = (height + 1) * 0.5

        K = np.zeros((3, 3))
        K[0, 0] = 1 / psx
        K[0, 2] = Ox
        K[1, 1] = - (1 / psy)
        K[1, 2] = Oy
        K[2, 2] = 1

        print(K)
        '''

        return RT

    def __getitem__(self, idx):
        """

        :param idx: item to choose
        :return: dictionary with following format:
            {
                'image': image,
                'depth': depth,
                'cam': cam,
                'output': output
            }
            where
              image is a WxHxC matrix of floats,
              depth is a WxH matrix of floats,
              cam is a dictionary:
                {
                    'RT1': RT1,
                    'RT2': RT2,
                    'K': ICLNUIMDataset.K
                }
                where
                  RT1 is a 3x4 extrinsic matrix of the idx-th item,
                  RT2 is a 3x4 extrinsic matrix of a random neighboring item or None (see self.sampleOutput)
                  K is a 3x3 intrinsic matrix (constant over all items),
              output is a dictionary or None (see self.sampleOutput):
                {
                  'image': output_image,
                  'idx': output_idx
                }
                where
                  image is a random neighboring image
                  idx is the index of the neighboring image (and of cam['RT2'])



        """
        image = self.load_image(idx)
        depth = self.load_depth(idx)

        RT1 = self.load_cam(idx)
        cam = {
            'RT1': RT1,
            'K': ICLNUIMDataset.K
        }

        output = None
        if self.sampleOutput:
            # sample second idx in [idx-30, idx+30) interval
            low = idx - 30 if idx >= 30 else 0
            high = idx + 30 if idx <= self.size - 30 else self.size
            output_idx = np.random.randint(low, high, 1)[0] # high is exclusive

            # never return the same idx, default handling: just use +1 or -1 idx
            if output_idx == idx and self.size > 1: # if we only have one sample, we can do nothing about this.
                output_idx = idx+1 if idx < self.size-1 else idx-1

            output_idx = idx + 1 if idx < self.size - 1 else idx # todo remove

            # load image of new index
            output_image = self.load_image(output_idx)

            # load cam of new index
            RT2 = self.load_cam(output_idx)
            cam['RT2'] = RT2

            output = {
                'image': output_image,
                'idx': output_idx
            }

        if self.cam_transforms:
            cam['K'], cam['Kinv'] = transform_matrices(cam['K'], isK=True)
            cam['RT1'], cam['RT1inv'] = transform_matrices(cam['RT1'])
            cam['RT2'], cam['RT2inv'] = transform_matrices(cam['RT2'])


        sample = {
            'image': image,
            'depth': depth,
            'cam': cam,
            'output': output
        }

        if self.transform:
            sample['image'] = self.transform(sample['image'])
            sample['depth'] = self.transform(sample['depth'])
            if self.sampleOutput:
                sample['output']['image'] = self.transform(sample['output']['image'])

        return sample

    def __len__(self):
        return self.size

    def toImagePlane(self, depth, x, y):
        # taken from the c++ code implementation at https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html in file VaFRIC.cpp#getEuclidean2PlanarDepth
        x_plane = (x - ICLNUIMDataset.cam_K['cx']) / ICLNUIMDataset.cam_K['fx']
        y_plane = (y - ICLNUIMDataset.cam_K['cy']) / ICLNUIMDataset.cam_K['fy']
        return depth / np.sqrt(x_plane ** 2 + y_plane ** 2 + 1)


def test():
    dataset = ICLNUIMDataset("/home/lukas/Desktop/datasets/ICL-NUIM/prerendered_data/living_room_traj0_loop",
                             depth_to_image_plane=True,
                             sampleOutput=True)
    #dataset = ICLNUIMDataset("sample", depth_to_image_plane=True, sampleOutput=True);

    print("Length of dataset: {}".format(len(dataset)))

    # Show first item in the dataset
    i = 0
    item = dataset.__getitem__(i)

    print("RT1:\n{}". format(item['cam']['RT1']))
    print("RT2:\n{}".format(item['cam']['RT2']))
    print("K:\n{}".format(item['cam']['K']))

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=item['depth'].shape)
    fig.suptitle("Sample " + str(i), fontsize=16)
    img = item['image']
    depth = item['depth']
    out_img = item['output']['image']
    out_idx = item['output']['idx']
    fig.add_subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(img)
    fig.add_subplot(1, 3, 2)
    plt.title("Depth Map")
    plt.imshow(depth, cmap='gray')
    fig.add_subplot(1, 3, 3)
    plt.title("Output Image " + str(out_idx))
    plt.imshow(out_img)

    plt.show()


if __name__ == "__main__":
    # execute only if run as a script
    test()
