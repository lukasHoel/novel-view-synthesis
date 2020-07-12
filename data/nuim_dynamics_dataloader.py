from util.camera_transformations import *
import torchvision

from data.nuim_dataloader import ICLNUIMDataset
import os
import json


class ICLNUIM_Dynamic_Dataset(ICLNUIMDataset):

    def __init__(self,
                 path,
                 input_as_segmentation=True,
                 sampleOutput=True,
                 output_from_other_view=False,
                 inverse_depth=False,
                 cacheItems=False,
                 transform=None,
                 out_shape=(480,640)):
        self.output_from_other_view = output_from_other_view
        self.input_as_segmentation = input_as_segmentation

        ICLNUIMDataset.__init__(self,
                             path=path,
                             sampleOutput=sampleOutput,
                             inverse_depth=inverse_depth,
                             cacheItems=cacheItems,
                             transform=transform,
                             out_shape=out_shape)

    def load_data(self, dir_content):

        # Load similar to Matterport: all "original" are in beginning of list, all "moved" in end of list and return size == len(img) // 2
        # Duplicate the dynamics file across each index because it is similar for all images
        # return None depth and fix in disk_dataloader to allow None depth
        # cam can be duplicated to be of size len(img) because load_ext_cam must work for out_image as well.

        # load originals
        img_seg = sorted([os.path.join("original", f) for f in os.listdir(os.path.join(self.path, "original")) if f.endswith(".seg.png")])
        if self.input_as_segmentation:
            # segmentation images are also the ones that should be used as input in the dataset samples
            img = img_seg
        else:
            # segmentation images are only needed for calculating dynamic mask. The input images should be the rgb images.
            img = sorted([os.path.join("original", f) for f in os.listdir(os.path.join(self.path, "original")) if f.endswith(".png") and not f.endswith(".seg.png")])

        cam = sorted([os.path.join("original", f) for f in os.listdir(os.path.join(self.path, "original")) if f.endswith(".txt")])
        size = len(img)
        depth = sorted([os.path.join("original", f) for f in os.listdir(os.path.join(self.path, "original")) if f.endswith('.gl.depth')])
        has_depth = len(depth) > 0
        depth_binary = sorted([os.path.join("original", f) for f in os.listdir(os.path.join(self.path, "original")) if f.endswith('.gl.depth.npy')])
        has_binary_depth = len(depth_binary) > 0

        # load moved img
        moved_img = sorted([os.path.join("moved", f) for f in os.listdir(os.path.join(self.path, "moved")) if f.endswith(".seg.png")])
        if len(moved_img) != len(img):
            raise ValueError("number of .png files in 'original' ({}) and 'moved' ({}) not identical".format(len(img), len(moved_img)))
        else:
            img.extend(moved_img)
            img.extend(moved_img)

        # load moved depth and depth.npy
        moved_depth = sorted([os.path.join("moved", f) for f in os.listdir(os.path.join(self.path, "moved")) if f.endswith('.gl.depth')])
        if len(moved_depth) != len(depth):
            raise ValueError("number of .depth files in 'original' ({}) and 'moved' ({}) not identical".format(len(depth), len(moved_depth)))
        else:
            depth.extend(moved_depth)

        moved_depth_binary = sorted([os.path.join("moved", f) for f in os.listdir(os.path.join(self.path, "moved")) if f.endswith('.gl.depth.npy')])
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
            dynamics = dynamics[0] # TODO SUPPORT MULTIPLE TRANSFORMATIONS IN ONE JSON

        return img, depth, has_depth, depth_binary, has_binary_depth, cam, size, img_seg, dynamics

    def modify_dynamics_transformation(self, transformation):
        """
        In ICL-DYNAMICS we render with a c++ renderer that applies a SCALE_X(-1) before applying the RT matrices.
        But we moved vertices of the mesh before applying the SCALE_X(-1) matrix which is why we need to manually simulate this here.
        Thus, this code makes the necessary changes to the RT matrix to take this into account.
        """
        transformation = np.copy(transformation) # copy to not modify the original array for subsequent calls

        # MODIFY TRANSLATION: SCALE_X(-1) flips the translation along the x-axis
        transformation[0,3] *= -1

        # MODIFY ROTATION: SCALE_X(-1) negates the rotation angles of y and z axis
        R = transformation[0:3, 0:3]
        theta = rotationMatrixToEulerAngles(R)
        theta[1] *= -1
        theta[2] *= -1
        R = eulerAnglesToRotationMatrix(theta)
        transformation[0:3, 0:3] = R

        return transformation

    def create_input_to_output_sample_map(self):
        # since we have all originals followed by all moved in the self.img list and size == len(originals), we get the associated moved file from the original file by adding an index of size
        if not self.output_from_other_view:
            # just return moved image from same camera
            return [idx + self.size for idx in range(self.size)]
        else:
            # return moved image with idx + 1, so from the next camera (except for last image, here return idx - 1 moved image)
            return [idx + self.size + 1 if idx < self.size - 1 else idx + self.size - 1 for idx in range(self.size)]

    def modify_depth(self, depth):
        return depth
        # nothing to do here because we expect the dynamic images to be rendered by our custom renderer which produces already the correct depth values
        # a test has shown that with doing it like this, we get "straight walls" in world space, otherwise bent ones


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]
                    ])

    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]
                    ])

    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R

def test():

    size = 256

    transform = torchvision.transforms.Compose([
        #torchvision.transforms.Resize((size, size)),
        torchvision.transforms.ToTensor(),
    ])

    dataset = ICLNUIM_Dynamic_Dataset("/home/lukas/Desktop/datasets/ICL-NUIM/custom/seq0003",
                             input_as_segmentation=True,
                             sampleOutput=True,
                             output_from_other_view=False,
                             inverse_depth=False,
                             cacheItems=False,
                             transform=transform)
    #dataset = ICLNUIMDataset("sample", sampleOutput=True, transform=transform);

    print("Length of dataset: {}".format(len(dataset)))

    # Show first item in the dataset
    i = 0
    item = dataset.__getitem__(i)

    print(item["image"].shape)
    print(item["dynamics"]["input_mask"].shape)
    print(item["dynamics"]["output_mask"].shape)
    print(item["dynamics"]["transformation"])

    print("RT1:\n{}". format(item['cam']['RT1']))
    print("R1 euler angles in radians: {}".format(rotationMatrixToEulerAngles(item['cam']['RT1'].cpu().numpy()[0:3, 0:3])))
    print("RT2:\n{}".format(item['cam']['RT2']))
    print("R2 euler angles in radians: {}".format(rotationMatrixToEulerAngles(item['cam']['RT2'].cpu().numpy()[0:3, 0:3])))
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

    fig.add_subplot(1, 5, 1)
    plt.title("Image")
    plt.imshow(img)

    fig.add_subplot(1, 5, 2)
    plt.title("Output Image " + str(out_idx))
    plt.imshow(out_img)

    fig.add_subplot(1, 5, 3)
    plt.title("Mask dynamics at input")
    img[:,:] = np.array([0, 0, 0])
    mask = np.moveaxis(item["dynamics"]["input_mask"].numpy(), 0, -1).squeeze()
    img[mask == 1] = np.array([1, 1, 1])
    plt.imshow(img)

    fig.add_subplot(1, 5, 4)
    plt.title("Mask dynamics at output")
    img[:, :] = np.array([0, 0, 0])
    mask = np.moveaxis(item["dynamics"]["output_mask"].numpy(), 0, -1).squeeze()
    img[mask == 1] = np.array([1, 1, 1])
    plt.imshow(img)

    fig.add_subplot(1, 5, 5)
    plt.title("Input Depth Map")
    plt.imshow(depth)
    plt.show()

if __name__ == "__main__":
    # execute only if run as a script
    test()
