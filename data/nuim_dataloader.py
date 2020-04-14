import numpy as np
import os
import torch
from torch.utils.data import Dataset
from skimage import io


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

    def __init__(self, path, depth_to_image_plane=True, transform=None):
        '''

        :param path: path/to/NUIM/files. Needs to be a directory with .png, .depth and .txt files, as can be obtained from: https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html
        :param depth_to_image_plane: whether or not to convert to depth in the .depth file into image_plane depth, see: https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html
        :param transform: transform that should be applied to the input image AND the target depth
        '''
        self.transform = transform
        self.depth_to_image_plane = depth_to_image_plane
        self.path = path

        self.img = sorted([f for f in os.listdir(path) if f.endswith('.png')])
        self.depth = sorted([f for f in os.listdir(path) if f.endswith('.depth')])
        self.cam = sorted([f for f in os.listdir(path) if f.endswith('.txt')])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # read image
        image = io.imread(os.path.join(self.path, self.img[idx]))

        # read depth file
        with open(os.path.join(self.path, self.depth[idx])) as f:
            depth = [float(i) for i in f.read().split(' ') if i.strip()] # read .depth file
            depth = np.asarray(depth, dtype=np.float32).reshape(image.shape[:2]) # convert to same format as image WxH

            # convert to image plane depth by taking into account the position in the WxH array as (x, y)
            if self.depth_to_image_plane:
                depth = np.fromfunction(lambda x, y: self.toImagePlane(depth, x, y), depth.shape, dtype=depth.dtype)

        #TODO: read cam.txt file

        sample = {'image': image,
                  'depth': depth}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
            sample['depth'] = self.transform(sample['depth'])

        return sample

    def __len__(self):
        return len(self.img)

    def toImagePlane(self, depth, x, y):
        # taken from the c++ code implementation at https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html in file VaFRIC.cpp#getEuclidean2PlanarDepth
        x_plane = (x - ICLNUIMDataset.cam_K['cx']) / ICLNUIMDataset.cam_K['fx']
        y_plane = (y - ICLNUIMDataset.cam_K['cy']) / ICLNUIMDataset.cam_K['fy']
        return depth / np.sqrt(x_plane**2 + y_plane**2 + 1)

def test():
    #dataset = ICLNUIMDataset("/home/lukas/ICL-NUIM/prerendered_data/living_room_traj0_loop", depth_to_image_plane=True);
    dataset = ICLNUIMDataset("sample", depth_to_image_plane=True);

    print("Length of dataset: {}".format(len(dataset)))

    # Show first item in the dataset
    item = dataset.__getitem__(0)

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=item['depth'].shape)
    fig.suptitle("First sample", fontsize=16)
    img = item['image']
    depth = item['depth']
    fig.add_subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(img)
    fig.add_subplot(1, 2, 2)
    plt.title("Depth Map")
    plt.imshow(depth)

    plt.show()


if __name__ == "__main__":
    # execute only if run as a script
    test()