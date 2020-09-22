import torch
import numpy as np


seg = torch.randint(0, 10, (256, 256))

def randomize(segmentation, num_classes=10, movement_id=None, scene=None, seed=13):

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    classes = np.arange(num_classes+1)
    np.random.shuffle(classes)

    result = torch.zeros_like(segmentation)

    if movement_id is not None:
        replacement = classes[movement_id]
        result[segmentation==movement_id] = replacement
        segmentation[segmentation==movement_id] = 0
        result[segmentation==replacement] = movement_id
        segmentation[segmentation==replacement] = 0

        result += segmentation

        return result

    else:
        for i in range(num_classes+1):
            mask = segmentation == i
            replacement = classes[i]
            result[mask] = replacement

        return result

