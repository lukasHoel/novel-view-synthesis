from util.camera_transformations import *
import torchvision

from data.disk_dataloader import DiskDataset
import os
import re

class MP3D_Habitat_Offline_Dataset(DiskDataset):
    '''
    Loads samples from the Matterport3D dataset via Habitat from disk.
    '''

    # intrinsic camera matrix
    cam_K = {
        'fx': 1.0,
        'fy': 1.0,
        'cx': 0.0,
        'cy': 0.0
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

    # max depth that we accept for MP3D - can be bigger but we cut it at this value
    max_depth = 10.0

    def __init__(self,
                 path,
                 in_size,
                 sampleOutput=True,
                 inverse_depth=False,
                 cacheItems=False,
                 transform=None):
        '''

        :param path: path/to/NUIM/files. Needs to be a directory with .png, .depth and .txt files, as can be obtained from: https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html
        :param sampleOutput: whether or not to sample the second image from disk. If idx points to "_0" image, the "_1" image is returned, otherwise it is the other way round.
                For example: If the 500. item is accessed, the second camera pose (R|T) will be from any of the poses of the items 470-530 (excluding 500).
        :param inverse_depth: If true, depth.pow(-1) is returned for the depth file (changing depth BEFORE applying transform object).
        :param transform: transform that should be applied to the input image AND the target depth
        :param in_size: size of rectangular images read from disk
        '''
        DiskDataset.__init__(self,
                             path=path,
                             maxDepth=MP3D_Habitat_Offline_Dataset.max_depth,
                             imageInputShape=(in_size, in_size),
                             sampleOutput=sampleOutput,
                             inverse_depth=inverse_depth,
                             cacheItems=cacheItems,
                             transform=transform)

    def parse_directories(self):
        folder_names = os.listdir(self.path)
        full_folder_paths = map(lambda x: os.path.join(self.path, x), folder_names)
        files = []
        for i, folder in enumerate(full_folder_paths):
            for file in os.listdir(folder):
                files.append(os.path.join(folder_names[i], file))
        files = sorted(files)

        self.img = list(filter(lambda x: x.endswith('.png'), files))
        self.depth = list(filter(lambda x: x.endswith('.depth'), files))
        self.depth_binary = list(filter(lambda x: x.endswith('.depth.npy'), files))
        self.has_binary_depth = len(self.depth_binary) > 0
        self.cam = list(filter(lambda x: x.endswith('.txt'), files))

    def modify_depth(self, depth):
        return depth # nothing to do here

    def load_int_cam(self):
        return MP3D_Habitat_Offline_Dataset.K, MP3D_Habitat_Offline_Dataset.Kinv

    def create_input_to_output_sample_map(self):
        regex = r"\w+_\d+_(\d+).\w+"
        inputToOutputIndex = []
        
        for idx in range(self.size):
            pair_id = int(re.search(regex, self.img[idx]).group(1))
            if pair_id == 0:
                inputToOutputIndex.append(idx+1)
            else:
                inputToOutputIndex.append(idx-1)
            # NOTE: n > 2 views, current implementation should be changed

        return inputToOutputIndex


def getEulerAngles(R):
    ry = np.arcsin(R[0,2])
    rz = np.arccos(R[0,0] / np.cos(ry))
    rx = np.arccos(R[2,2] / np.cos(ry))

    return rx, ry, rz


def test():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((512, 512)),
        torchvision.transforms.ToTensor(),
    ])

    dataset = MP3D_Habitat_Offline_Dataset("./data/mp3d_dataset/",
                             in_size=256,
                             sampleOutput=True,
                             inverse_depth=False,
                             cacheItems=False,
                             transform=transform)

    print("Length of dataset: {}".format(len(dataset)))

    # Show first item in the dataset
    import numpy as np
    i = np.random.randint(len(dataset))
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
    plt.title("Input Image")
    plt.imshow(img)
    fig.add_subplot(1, 3, 2)
    plt.title("Input Depth Map")
    plt.imshow(depth)
    fig.add_subplot(1, 3, 3)
    plt.title("Output Image")
    plt.imshow(out_img)

    plt.show()



if __name__ == "__main__":
    # execute only if run as a script
    test()
