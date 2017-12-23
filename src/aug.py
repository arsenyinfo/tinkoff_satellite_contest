import numpy as np
from skimage.transform import rotate, resize


def split(sandwich, n_layers):
    return sandwich[:, :, :n_layers], sandwich[:, :, n_layers:]


def augment(patch, target, stretched_size):
    if np.random.rand() > .5:
        if np.random.rand() > .5:
            patch, target = map(lambda x: np.fliplr(x), (patch, target))
        if np.random.rand() > .5:
            patch, target = map(lambda x: np.flipud(x), (patch, target))
        if np.random.rand() > .75:
            patch, target = map(lambda x: np.rot90(x, 1), (patch, target))
        if np.random.rand() > .75:
            patch, target = map(lambda x: np.rot90(x, 3), (patch, target))
        if np.random.rand() > .75:
            patch, target = map(lambda x: rotate(x, angle=np.random.randint(5, 80), mode='reflect'),
                                (patch, target))
    else:
        sandwich = np.dstack((patch, target))
        expected_shape = patch.shape, target.shape

        if np.random.rand() > .5:
            sandwich = rotate(sandwich, np.random.randint(0, 360), mode='reflect')
        else:
            patch_size = patch.shape[0]
            i = (stretched_size - patch_size) // 2
            j = stretched_size - i
            if np.random.rand() > .5:
                sandwich = resize(sandwich, (stretched_size, stretched_size))
                sandwich = sandwich[i: j, i: j, :]
            else:
                if np.random.rand() > .5:
                    sandwich = resize(sandwich, (stretched_size, patch_size))
                    sandwich = sandwich[i: j, :, :]
                else:
                    sandwich = resize(sandwich, (patch_size, stretched_size))
                    sandwich = sandwich[:, i: j, :]

        patch, target = split(sandwich, n_layers=patch.shape[-1])
        assert patch.shape, target.shape == expected_shape

    return patch, target