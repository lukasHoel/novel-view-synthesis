import os
import numpy as np

from tqdm.auto import tqdm

def load_depth(file, img_shape=(480, 640)):
    with open(file) as f:
        depth = [float(i) for i in f.read().split(' ') if i.strip()]  # read .depth file
        depth = np.asarray(depth, dtype=np.float32).reshape(img_shape)  # convert to same format as image WxH
    return depth

def main(input, output, shape):
    depth_files = sorted([os.path.join(input, f) for f in os.listdir(input) if f.endswith('.depth')])

    print("Converting {} depth files".format(len(depth_files)))

    for file in tqdm(depth_files):
        depth = load_depth(file, shape)
        out_path = os.path.join(output, file + ".npy") # actively keep extension and only add .npy to it, e.g. foo.depth.npy from foo.depth
        np.save(out_path, depth)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert .depth text files float files to binary numpy .npy files')
    parser.add_argument('--input', metavar='pathToFile', required=True,
                        help='path/to/input/directory')
    parser.add_argument('--output', metavar='pathToDirectory', required=False, default=None,
                        help='path/to/output/directory. Optional, default: input directory')
    parser.add_argument('--width', metavar='N', type=int, required=False, default=640,
                        help='image width')
    parser.add_argument('--height', metavar='N', type=int, required=False, default=480,
                        help='image width')

    args = parser.parse_args()
    main(input=args.input,
         output=args.output if args.output is not None else args.input,
         shape=(args.height, args.width))