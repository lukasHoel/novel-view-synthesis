import os
from torch.utils.data import Dataset
from PIL import Image
from util.camera_transformations import *
import re
import torchvision

from abc import ABC, abstractmethod


class ToNumpy(object):
    def __call__(self, sample):
        return np.array(sample)


class ClipDepth(object):
    '''Set maximal depth'''

    def __init__(self, maxDepth):
        self.maxDepth = maxDepth

    def __call__(self, sample):
        sample[sample>self.maxDepth] = self.maxDepth
        return sample


class DiskDataset(Dataset, ABC):
    '''
    Loads samples from files satisfying the following directory structure:
    TODO describe
    '''

    # regex to read lines from the camera .txt file
    cam_pattern = "(?P<id>.*\w).*= \[(?P<x>.*), (?P<y>.*), (?P<z>.*)\].*"

    def __init__(self,
                 path,
                 maxDepth,
                 imageInputShape,
                 sampleOutput=True,
                 inverse_depth=False,
                 cacheItems=False,
                 transform=None):
        '''

        :param path: path/to/<base>/files. Needs to be a directory with .png, .depth and .txt files
        :param maxDepth: maximum depth that is accepted. Everything above that will be cut to given value.
        :param imageInputShape: the original image input shape for the dataset that gets used. Necessary to reshape depth values into correct array dimensions.
        :param sampleOutput: whether or not to sample an output image. Implementations specify how the output is retrieved.
        :param inverse_depth: If true, depth.pow(-1) is returned for the depth file (changing depth BEFORE applying transform object).
        :param cacheItems: If true, all items will be stored in RAM dictionary, once they were accessed.
        :param transform: transform that should be applied to the input image AND the target depth
        '''
        self.transform = transform
        self.K, self.Kinv = self.load_int_cam()
        self.imageInputShape = imageInputShape

        # Fix for this issue: https://github.com/pytorch/vision/issues/2194
        if isinstance(self.transform.transforms[-1], torchvision.transforms.ToTensor):
            self.transform_depth = torchvision.transforms.Compose([
                *self.transform.transforms[:-1],
                ToNumpy(),
                ClipDepth(maxDepth),
                torchvision.transforms.ToTensor()
            ])
        else:
            self.transform_depth = self.transform

        self.inverse_depth = inverse_depth
        self.maxDepth = maxDepth

        self.path = path
        self.img = sorted([f for f in os.listdir(path) if f.endswith('.png')])
        self.depth = sorted([f for f in os.listdir(path) if f.endswith('.depth')])
        self.depth_binary = sorted([f for f in os.listdir(path) if f.endswith('.depth.npy')])
        self.has_binary_depth = len(self.depth_binary) > 0
        self.cam = sorted([f for f in os.listdir(path) if f.endswith('.txt')])

        self.size = len(self.img)

        self.sampleOutput = sampleOutput
        if self.sampleOutput:
            self.inputToOutputIndex = self.create_input_to_output_sample_map()

        self.cacheItems = cacheItems
        self.itemCache = [None for i in range(self.size)]

    def load_image(self, idx):
        return Image.open(os.path.join(self.path, self.img[idx]))

    def load_depth(self, idx):
        if self.has_binary_depth:
            # read faster from .depth.npy file - this is much faster than parsing the char-based .depth file from ICL directly.
            depth = np.load(os.path.join(self.path, self.depth_binary[idx]))
        else:
            with open(os.path.join(self.path, self.depth[idx])) as f:
                depth = [float(i) for i in f.read().split(' ') if i.strip()]  # read .depth file
                depth = np.asarray(depth, dtype=np.float32).reshape(self.imageInputShape)  # convert to same format as image HxW

        # Implementation specific
        depth = self.modify_depth(depth)

        # invert depth
        if self.inverse_depth:
            depth = np.power(depth, -1)

        return Image.fromarray(depth, mode='F') # return as float PIL Image

    def load_ext_cam(self, idx):
        cam = {} # load the .txt file in this dict
        with open(os.path.join(self.path, self.cam[idx])) as f:
            for line in f:
                m = re.match(DiskDataset.cam_pattern, line) # will match everything except angle, but that is not needed anyway
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

        # combine in correct shape
        RT = np.column_stack((x, y, z, cam["cam_pos"]))
        RT = np.vstack([RT, [0,0,0,1]])
        RT = RT.astype(np.float32)

        # calculate RTinv from RT
        RTinv = np.linalg.inv(RT).astype(np.float32)

        # return as torch tensor
        RT = torch.from_numpy(RT)
        RTinv = torch.from_numpy(RTinv)

        return RT, RTinv

    @abstractmethod
    def load_int_cam(self):
        pass

    @abstractmethod
    def modify_depth(self, depth):
        pass

    @abstractmethod
    def create_input_to_output_sample_map(self):
        pass

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
                    'RT2inv': RT2inv,
                    'K': ICLNUIMDataset.K,
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
        depth = self.load_depth(idx)
        RT1, RT1inv = self.load_ext_cam(idx)

        cam = {
            'RT1': RT1,
            'RT1inv': RT1inv,
            'K': self.K,
            'Kinv': self.Kinv
        }

        output = None
        if self.sampleOutput:
            # lookup output for this sample
            output_idx = self.inputToOutputIndex[idx]

            # load image of new index
            output_image = self.load_image(output_idx)
            output_depth = self.load_depth(output_idx)

            # load cam of new index
            RT2, RT2inv = self.load_ext_cam(output_idx)

            cam['RT2'] = RT2
            cam['RT2inv'] = RT2inv

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
