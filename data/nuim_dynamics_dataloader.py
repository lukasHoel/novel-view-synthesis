from util.camera_transformations import *
import torchvision

from data.nuim_dataloader import ICLNUIMDataset
import os
import json


class ICLNUIM_Dynamic_Dataset(ICLNUIMDataset):

    def __init__(self,
                 path,
                 sampleOutput=True,
                 output_from_other_view=False,
                 inverse_depth=False,
                 cacheItems=False,
                 transform=None):
        self.output_from_other_view = output_from_other_view

        ICLNUIMDataset.__init__(self,
                             path=path,
                             sampleOutput=sampleOutput,
                             inverse_depth=inverse_depth,
                             cacheItems=cacheItems,
                             transform=transform)

    def load_data(self, dir_content):

        # Load similar to Matterport: all "original" are in beginning of list, all "moved" in end of list and return size == len(img) // 2
        # Duplicate the dynamics file across each index because it is similar for all images
        # return None depth and fix in disk_dataloader to allow None depth
        # cam can be duplicated to be of size len(img) because load_ext_cam must work for out_image as well.

        # load originals
        img = sorted([os.path.join("original", f) for f in os.listdir(os.path.join(self.path, "original")) if f.endswith(".png") and not f.endswith(".depth.png")])
        cam = sorted([os.path.join("original", f) for f in os.listdir(os.path.join(self.path, "original")) if f.endswith(".txt")])
        size = len(img)
        depth = sorted([os.path.join("original", f) for f in os.listdir(os.path.join(self.path, "original")) if f.endswith('.depth')])
        has_depth = len(depth) > 0
        depth_binary = sorted([os.path.join("original", f) for f in os.listdir(os.path.join(self.path, "original")) if f.endswith('.depth.npy')])
        has_binary_depth = len(depth_binary) > 0

        # load moved img
        moved_img = sorted([os.path.join("moved", f) for f in os.listdir(os.path.join(self.path, "moved")) if f.endswith(".png") and not f.endswith(".depth.png")])
        if len(moved_img) != len(img):
            raise ValueError("number of .png files in 'original' ({}) and 'moved' ({}) not identical".format(len(img), len(moved_img)))
        else:
            img.extend(moved_img)

        # load moved depth and depth.npy
        moved_depth = sorted([os.path.join("moved", f) for f in os.listdir(os.path.join(self.path, "moved")) if f.endswith('.depth')])
        if len(moved_depth) != len(depth):
            raise ValueError("number of .depth files in 'original' ({}) and 'moved' ({}) not identical".format(len(depth), len(moved_depth)))
        else:
            depth.extend(moved_depth)

        moved_depth_binary = sorted([os.path.join("moved", f) for f in os.listdir(os.path.join(self.path, "moved")) if f.endswith('.depth.npy')])
        if len(moved_depth_binary) != len(depth_binary):
            raise ValueError("number of .depth.npy files in 'original' ({}) and 'moved' ({}) not identical".format(len(depth_binary), len(moved_depth_binary)))
        else:
            depth_binary.extend(moved_depth_binary)

        # load moved cam: duplicate original cams
        cam.extend(cam.copy())

        # load dynamics
        dynamics_file = [f for f in dir_content if f == "moved.txt"][0]
        with open(os.path.join(self.path, dynamics_file)) as f:
            dynamics = json.load(f)

        return img, depth, has_depth, depth_binary, has_binary_depth, cam, size, dynamics

    def create_input_to_output_sample_map(self):
        # since we have all originals followed by all moved in the self.img list and size == len(originals), we get the associated moved file from the original file by adding an index of size
        if not self.output_from_other_view:
            # just return moved image from same camera
            return [idx + self.size for idx in range(self.size)]
        else:
            # return moved image with idx + 1, so from the next camera (except for last image, here return idx - 1 moved image)
            return [idx + self.size + 1 if idx < self.size - 1 else idx + self.size - 1 for idx in range(self.size)]

    #def modify_depth(self, depth):
        #return depth # nothing to do here because we expect the dynamic images to be rendered by our custom renderer which produces already the correct depth values

def getEulerAngles(R):
    ry = np.arcsin(R[0,2])
    rz = np.arccos(R[0,0] / np.cos(ry))
    rx = np.arccos(R[2,2] / np.cos(ry))

    return rx, ry, rz


def test():

    size = 256

    transform = torchvision.transforms.Compose([
        #torchvision.transforms.Resize((size, size)),
        torchvision.transforms.ToTensor(),
    ])

    dataset = ICLNUIM_Dynamic_Dataset("/home/lukas/Desktop/datasets/ICL-NUIM/custom/seq0001",
                             sampleOutput=True,
                             output_from_other_view=True,
                             inverse_depth=False,
                             cacheItems=False,
                             transform=transform)
    #dataset = ICLNUIMDataset("sample", sampleOutput=True, transform=transform);

    print("Length of dataset: {}".format(len(dataset)))

    # Show first item in the dataset
    i = 5
    item = dataset.__getitem__(i)

    print(item["image"].shape)
    print(item["dynamics"]["mask"].shape)
    print(item["dynamics"]["transformation"])

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
    out_img = np.moveaxis(item['output']['image'].numpy(), 0, -1)
    out_idx = item['output']['idx']

    depth = np.moveaxis(item['depth'].numpy(), 0, -1).squeeze()
    print("MIN DEPTH", np.min(depth))
    print("MAX DEPTH", np.max(depth))

    fig.add_subplot(1, 4, 1)
    plt.title("Image")
    plt.imshow(img)

    fig.add_subplot(1, 4, 2)
    plt.title("Output Image " + str(out_idx))
    plt.imshow(out_img)

    fig.add_subplot(1, 4, 3)
    plt.title("Mask dynamics")
    img[:,:] = np.array([0, 0, 0])
    mask = np.moveaxis(item["dynamics"]["mask"].numpy(), 0, -1).squeeze()
    img[mask == 1] = np.array([1, 1, 1])
    plt.imshow(img)

    fig.add_subplot(1, 4, 4)
    plt.title("Input Depth Map")
    plt.imshow(depth, cmap='gray')
    plt.show()

    min_depth = -1
    max_depth = -1
    max_i = -1
    for i, item in enumerate(dataset):
        depth = np.moveaxis(item['depth'].numpy(), 0, -1).squeeze()
        min_d = np.min(depth)
        max_d = np.max(depth)
        if min_depth == -1 or min_d < min_depth:
            min_depth = min_d
        if max_depth == -1 or max_d > max_depth:
            max_depth = max_d
            max_i = i
    print("MIN DEPTH", min_depth)
    print("MAX DEPTH", max_depth)
    print("MAX INDEX", max_i)

if __name__ == "__main__":
    # execute only if run as a script
    test()
