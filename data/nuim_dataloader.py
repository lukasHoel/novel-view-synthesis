import numpy as np
import os
import torch
from torch.utils.data import Dataset
from skimage import io
from PIL import Image
from util.camera_transformations import *
import re
import torchvision

from time import time

class ToNumpy(object):
    def __call__(self, sample):
        return np.array(sample)

class ClipDepth(object):
    '''Set maximal depth'''

    def __call__(self, sample):
        sample[sample>10] = 10.0
        return sample

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

    # intrinsic camera matrix as torch tensor
    K = torch.from_numpy(np.zeros((4,4)).astype(np.float32))
    K[0,0] = cam_K['fx']
    K[1,1] = cam_K['fy']
    K[0,2] = cam_K['cx']
    K[1,2] = cam_K['cy']
    K[2,2] = 1
    K[3,3] = 1 # we use 4x4 matrix for easier backward-calculations without removing indices, see projection/z_buffer_manipulator.py

    # and inverted matrix as well
    Kinv = torch.from_numpy(np.zeros((4,4)).astype(np.float32))
    Kinv[:3,:3] = invert_K(K[:3,:3])
    Kinv[3,3] = 1

    # regex to read lines from the camera .txt file
    cam_pattern = "(?P<id>.*\w).*= \[(?P<x>.*), (?P<y>.*), (?P<z>.*)\].*"

    def __init__(self,
                 path,
                 depth_to_image_plane=True,
                 use_real_intrinsics=False,
                 sampleOutput=True,
                 RTrelativeToOutput=False,
                 inverse_depth=False,
                 cacheItems=False,
                 transform=None,
                 out_shape=(480,640)):
        '''

        :param path: path/to/NUIM/files. Needs to be a directory with .png, .depth and .txt files, as can be obtained from: https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html
        :param depth_to_image_plane: whether or not to convert to depth in the .depth file into image_plane depth, see: https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html
        :param sampleOutput: whether or not to uniformly sample a second image + extrinsic camera pose (R|T) in the neighborhood of each accessed item.
                neighborhood is currently defined as: select uniformly at random any camera in the range [index-30, index+30) where index is the accessed item index.
                For example: If the 500. item is accessed, the second camera pose (R|T) will be from any of the poses of the items 470-530 (excluding 500).
        :param RTrelativeToOutput: when sampleOutput=true, then this option will calculate relativ RT between cam1 and cam2 and return that as RT2. RT1 will be the identity.
        :param inverse_depth: If true, depth.pow(-1) is returned for the depth file (changing depth BEFORE applying transform object).
        :param transform: transform that should be applied to the input image AND the target depth
        :param use_real_intrinsics: If true, return the K and Kinv matrix from ICL dataset. If false return identity matrix.
        '''
        self.transform = transform
        if 'Resize' in str(transform):
            self.Resize = True
        else:
            self.Resize = False
        self.out_shape = out_shape
        # Fix for this issue: https://github.com/pytorch/vision/issues/2194
        if isinstance(self.transform.transforms[-1], torchvision.transforms.ToTensor):
            self.transform_depth = torchvision.transforms.Compose([
                *self.transform.transforms[:-1],
                ToNumpy(),
                ClipDepth(),
                torchvision.transforms.ToTensor()
            ])
        else:
            self.transform_depth = self.transform

        self.depth_to_image_plane = depth_to_image_plane
        self.use_real_intrinsics = use_real_intrinsics
        self.inverse_depth = inverse_depth
        self.path = path

        self.img = sorted([f for f in os.listdir(path) if f.endswith('.png')])
        self.depth = sorted([f for f in os.listdir(path) if f.endswith('.depth')])
        self.depth_binary = sorted([f for f in os.listdir(path) if f.endswith('.depth.npy')])
        self.has_binary_depth = len(self.depth_binary) > 0
        self.cam = sorted([f for f in os.listdir(path) if f.endswith('.txt')])

        self.size = len(self.img)

        self.sampleOutput = sampleOutput
        if self.sampleOutput:
            self.inputToOutputIndex = []
            for idx in range(self.size):
                # sample second idx in [idx-30, idx+30) interval
                low = idx - 30 if idx >= 30 else 0
                high = idx + 30 if idx <= self.size - 30 else self.size
                output_idx = np.random.randint(low, high, 1)[0] # high is exclusive

                # never return the same idx, default handling: just use +1 or -1 idx
                if output_idx == idx and self.size > 1: # if we only have one sample, we can do nothing about this.
                    output_idx = idx+1 if idx < self.size-1 else idx-1

                self.inputToOutputIndex.append(output_idx)

        self.RTrelativeToOutput = RTrelativeToOutput

        self.cacheItems = cacheItems
        self.itemCache = [None for i in range(self.size)]

    def load_image(self, idx):
        return Image.open(os.path.join(self.path, self.img[idx]))
        #return io.imread(os.path.join(self.path, self.img[idx]))

    def load_depth(self, idx, img_shape=(480, 640)):
        if self.has_binary_depth:
            # read faster from .depth.npy file - this is much faster than parsing the char-based .depth file from ICL directly.
            depth = np.load(os.path.join(self.path, self.depth_binary[idx]))
        else:
            with open(os.path.join(self.path, self.depth[idx])) as f:
                depth = [float(i) for i in f.read().split(' ') if i.strip()]  # read .depth file
                depth = np.asarray(depth, dtype=np.float32).reshape(img_shape)  # convert to same format as image WxH

        # convert to image plane depth by taking into account the position in the WxH array as (x, y)
        if self.depth_to_image_plane:
            # use (y,x) as input to lambda because the depth.shape is also in (y,x) format.
            depth = np.fromfunction(lambda y, x: self.toImagePlane(depth, x, y), depth.shape, dtype=depth.dtype)

        # invert depth
        if self.inverse_depth:
            depth = np.power(depth, -1)

        return Image.fromarray(depth, mode='F') # return as float PIL Image

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
        RT = np.vstack([RT, [0,0,0,1]]) # if (0 0 0 1) row is needed
        RT = RT.astype(np.float32)
        RTinv = np.linalg.inv(RT).astype(np.float32)
        #RTinv = np.vstack([RTinv, [0, 0, 0, 1]])  # if (0 0 0 1) row is needed

        # Code to calculate K - unnecessary because K is constant. taken from: https://www.doc.ic.ac.uk/~ahanda/VaFRIC/getcamK.m
        if self.Resize:
            K2 = torch.from_numpy(np.zeros((4,4)).astype(np.float32))
            K2[0,0] = 0.751875 * self.out_shape[1]      #cam_K2['fx']
            K2[1,1] = -1.0 * self.out_shape[0]          #cam_K2['fy']
            K2[0,2] = 0.5 * self.out_shape[1]           #cam_K2['cx']
            K2[1,2] = 0.5 * self.out_shape[0]           #cam_K2['cy']
            K2[2,2] = 1
            K2[3,3] = 1 # we use 4x4 matrix for easier backward-calculations without removing indices, see projection/z_buffer_manipulator.py

            K2inv = torch.from_numpy(np.zeros((4,4)).astype(np.float32))
            K2inv[:3,:3] = invert_K(K2[:3,:3])
            K2inv[3,3] = 1

            return RT, RTinv, K2, K2inv
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

        return RT, RTinv, ICLNUIMDataset.K, ICLNUIMDataset.Kinv

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
                    'RT1inv': RT1inv,
                    'RT2': RT2,
                    'RT2inv': RT2,inv
                    'K': ICLNUIMDataset.K
                    'Kinv': ICLNUIMDataset.Kinv
                }
                where
                  RT1 is a 4x4 extrinsic matrix of the idx-th item,
                  RT2 is a 4x4 extrinsic matrix of a random neighboring item or None (see self.sampleOutput)
                  K is a 4x4 intrinsic matrix (constant over all items) with 4th row/col added for convenience,
                  *inv is the inverted matrix
              output is a dictionary or None (see self.sampleOutput):
                {
                  'image': output_image,
                  'idx': output_idx
                }
                where
                  image is a random neighboring image
                  idx is the index of the neighboring image (and of cam['RT2'])



        """

        if self.itemCache[idx] is not None:
            return self.itemCache[idx]

        image = self.load_image(idx)
        #start = time()
        depth = self.load_depth(idx)
        #print("Depth loading took {}".format(time() - start))


        RT1, RT1inv, K, Kinv = self.load_cam(idx)

        cam = {
            'RT1': torch.from_numpy(RT1),
            'RT1inv': torch.from_numpy(RT1inv),
            'K': K if self.use_real_intrinsics else torch.eye(4),
            'Kinv': Kinv if self.use_real_intrinsics else torch.eye(4)
        }

        output = None
        if self.sampleOutput:
            # lookup output for this sample
            output_idx = self.inputToOutputIndex[idx]

            # load image of new index
            output_image = self.load_image(output_idx)
            output_depth = self.load_depth(output_idx)

            # load cam of new index
            RT2, RT2inv, K2, K2inv = self.load_cam(output_idx)

            if self.RTrelativeToOutput:
                #calculate relative RT matrix
                R1 = RT1[:3, :3]
                T1 = RT1[:3, 3]
                R2 = RT2[:3, :3]
                T2 = RT2[:3, 3]

                # RT
                #print(T1.shape)
                #print((T1-T2).shape)
                T = (R1.T@R2).dot(T2 - T1)
                RT = np.eye(4)
                RT[0:3, 0:3] = R1.T @ R2
                RT[:3, 3] = T
                RT = RT.astype(np.float32)
                RT = torch.from_numpy(RT)

                # RTinv
                #RTinv = np.linalg.inv(RT).astype(np.float32)
                RTinv = invert_RT(RT[:3,:])
                RTinv = np.vstack([RTinv, [0, 0, 0, 1]]).astype(np.float32)  # if (0 0 0 1) row is needed
                identity = torch.eye(4)
                RTinv = torch.from_numpy(RTinv)

                # Set dict
                cam['RT1'] = identity
                cam['RT1inv'] = identity
                cam['RT2'] = RT
                cam['RT2inv'] = RTinv
            else:
                cam['RT2'] = torch.from_numpy(RT2)
                cam['RT2inv'] = torch.from_numpy(RT2inv)

            output = {
                'image': output_image,
                'depth': output_depth,
                'idx': output_idx
            }


        sample = {
            'image': image,
            'depth': depth,
            'cam': cam,
            'output': output
        }

        if self.transform:
            sample['image'] = self.transform(sample['image'])
            sample['depth'] = self.transform_depth(sample['depth'])
            if self.sampleOutput:
                sample['output']['image'] = self.transform(sample['output']['image'])
                sample['output']['depth'] = self.transform_depth(sample['output']['depth'])

        if self.cacheItems:
            self.itemCache[idx] = sample

        return sample

    def __len__(self):
        return self.size

    def toImagePlane(self, depth, x, y):

        # taken from the figure in: https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html
        #z = ICLNUIMDataset.cam_K['fx'] * np.sqrt( (depth**2) / (x**2 + y**2 + ICLNUIMDataset.cam_K['fx']**2))


        # taken from the c++ code implementation at https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html in file VaFRIC.cpp#getEuclidean2PlanarDepth
        x_plane = (x - ICLNUIMDataset.cam_K['cx']) / ICLNUIMDataset.cam_K['fx']
        y_plane = (y - ICLNUIMDataset.cam_K['cy']) / ICLNUIMDataset.cam_K['fy']
        z = depth / np.sqrt(x_plane ** 2 + y_plane ** 2 + 1)

        return z

def getEulerAngles(R):
    ry = np.arcsin(R[0,2])
    rz = np.arccos(R[0,0] / np.cos(ry))
    rx = np.arccos(R[2,2] / np.cos(ry))

    return rx, ry, rz

def test():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
    ])

    dataset = ICLNUIMDataset("/home/lukas/Desktop/datasets/ICL-NUIM/prerendered_data/living_room_traj2_loop",
                             depth_to_image_plane=True,
                             use_real_intrinsics=False,
                             sampleOutput=True,
                             RTrelativeToOutput=False,
                             inverse_depth=False,
                             cacheItems=True,
                             transform=transform)
    #dataset = ICLNUIMDataset("sample", depth_to_image_plane=True, sampleOutput=True, transform=transform);

    print("Length of dataset: {}".format(len(dataset)))

    # Show first item in the dataset
    i = 400
    item = dataset.__getitem__(i)

    #print(item["depth"].numpy().flags)


    print(item["image"].shape)
    print(item["depth"].shape)

    print("RT1:\n{}". format(item['cam']['RT1']))
    print("RT1 euler angles in radians: {}".format(getEulerAngles(item['cam']['RT1'])))
    print("RT2:\n{}".format(item['cam']['RT2']))
    print("RT2 euler angles in radians: {}".format(getEulerAngles(item['cam']['RT2'])))
    print("K:\n{}".format(item['cam']['K']))

    print("RT1inv:\n{}". format(item['cam']['RT1inv']))
    print("RT2inv:\n{}".format(item['cam']['RT2inv']))
    print("Kinv:\n{}".format(item['cam']['Kinv']))

    print("K*Kinv:\n{}".format(item['cam']['K'].matmul(item['cam']['Kinv'])))
    print("RT1*RT1inv:\n{}".format(item['cam']['RT1'].matmul(item['cam']['RT1inv'])))
    print("RT2*RT2inv:\n{}".format(item['cam']['RT2'].matmul(item['cam']['RT2inv'])))

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=item['image'].shape[1:])
    fig.suptitle("Sample " + str(i), fontsize=16)
    img = np.moveaxis(item['image'].numpy(), 0, -1)
    depth = np.moveaxis(item['depth'].numpy(), 0, -1).squeeze()
    out_img = np.moveaxis(item['output']['image'].numpy(), 0, -1)
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
