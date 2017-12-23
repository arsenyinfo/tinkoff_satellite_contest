from glob import glob
from copy import copy

from tqdm import tqdm
import numpy as np
from scipy import signal
from keras.models import load_model
from skimage.io import imsave
from fire import Fire

from src.fit import parse_file
from src.utils import logger, FOLDS


class PatchCombiner:
    def __init__(self, target_shape, batch_size, window, overlay, x_data, predict, verbose=1):
        self.batch_size = batch_size
        self.window = window
        self.overlay = overlay
        self.i, self.j = 0, 0
        self.w, self.h, c = target_shape
        self.pred_mask, self.norm_mask = np.zeros(target_shape), np.zeros(target_shape)
        self.x_data = x_data
        self.predict = predict
        self.weights = self.get_weights()
        # this is assumed to be a keras method (or something with the same API)
        self.verbose = verbose
        self.counter = 0

    def get_patch(self):
        x = copy(self.i)
        y = copy(self.j)

        if x + self.window > self.w:
            x = self.w - self.window
        if y + self.window > self.h:
            y = self.h - self.window
        return self.x_data[x: x + self.window, y: y + self.window], (x, y)

    def _consume_batch(self, coords, batch, aug=None, deaug=None):
        for f in (aug, deaug):
            if f is not None and not callable(f):
                raise TypeError(f'aug and deaug should be both None or callable: aug {aug} and deaug {deaug} found')

        if aug is None and callable(deaug):
            raise TypeError(f'aug and deaug should be both None or callable: aug {aug} and deaug {deaug} found')
        if deaug is None and callable(aug):
            raise TypeError(f'aug and deaug should be both None or callable: aug {aug} and deaug {deaug} found')

        if aug is not None and deaug is not None:
            predicted_batch = deaug(self.predict(aug(batch), verbose=self.verbose))
        else:
            predicted_batch = self.predict(batch, verbose=self.verbose)

        for (x, y), pred in zip(coords, predicted_batch):
            self.pred_mask[x:x + self.window, y:y + self.window] += pred * self.weights
            self.norm_mask[x:x + self.window, y:y + self.window] += self.weights

    def consume_batch(self, coords, batch):
        self.counter += len(batch)

        batch = np.array(batch)

        self._consume_batch(coords, np.copy(batch))
        self._consume_batch(coords, np.copy(batch), aug=np.fliplr, deaug=np.fliplr)
        self._consume_batch(coords, np.copy(batch), aug=np.flipud, deaug=np.flipud)

        coords, batch = [], []
        return coords, batch

    def get_weights(self):
        gkern1d = signal.gaussian(self.window, self.window / 3).reshape(self.window, 1)
        gkern2d = np.outer(gkern1d, gkern1d)
        w = np.expand_dims(gkern2d, axis=-1)
        return w

    def process(self, png_ready=True):
        coords, batch = [], []

        while self.i <= self.w and self.j <= self.h:
            if len(batch) < self.batch_size:
                patch, coord = self.get_patch()
                batch.append(patch)
                coords.append(coord)

                self.i += self.overlay
                if self.i > self.w:
                    self.i = 0
                    self.j += self.overlay

            else:
                coords, batch = self.consume_batch(coords, batch)

        self.consume_batch(coords, batch)
        if self.verbose:
            logger.info(f"The image was combined from {self.counter} patches")

        if png_ready:
            img = (self.pred_mask / self.norm_mask) * 255.
            img = img.astype('uint8')
            return img
        return self.pred_mask / self.norm_mask


def test_patch_combiner():
    def predict(batch, verbose=1):
        return np.ones(batch.shape)

    x_data = np.random.rand(1000, 1000, 3)
    tile = PatchCombiner(target_shape=(1000, 1000, 3), batch_size=64, window=100, overlay=50, x_data=x_data,
                         predict=predict, verbose=1)
    img = tile.process()

    eps = .0001
    assert 1 + eps > img.mean() > 1 + eps
    assert 1 + eps > img.min() > 1 + eps
    assert 1 + eps > img.max() > 1 + eps


def plot_all(mask):
    folds = FOLDS
    models = glob(f'models/{mask}*')

    test_imgs_ids = [x.split('/')[-1].split('.')[0] for x in glob('data/tif/tif_test/*.tif')]
    test = {x: None for x in test_imgs_ids}
    logger.info(f'There are {len(test)} images in test')

    for model_name in models:
        n_fold = int(model_name.split('.')[0].split('_')[1])
        model = load_model(model_name, compile=False)
        logger.info(f'Processing fold {n_fold}')

        for id_ in tqdm(folds[n_fold], desc='train images'):
            x_data, y_data = parse_file(id_)

            tile = PatchCombiner(target_shape=y_data.shape,
                                 batch_size=64,
                                 window=256,
                                 overlay=64,
                                 x_data=x_data,
                                 predict=model.predict,
                                 verbose=0)

            img = tile.process()
            img = np.dstack((np.zeros(x_data.shape).astype('uint8'), img))
            imsave(f'predicts/train/{id_}_proba.png', img)

        for id_ in tqdm(test, desc='test images'):
            x_data = parse_file(id_, is_train=False)
            tile = PatchCombiner(target_shape=y_data.shape,
                                 batch_size=64,
                                 window=256,
                                 overlay=64,
                                 x_data=x_data,
                                 predict=model.predict,
                                 verbose=0)

            img = tile.process()
            img = np.dstack((np.zeros(x_data.shape), img))

            if test[id_] is None:
                test[id_] = img
            else:
                test[id_] += img

    for id_, img in test.items():
        img = (img / len(folds)).astype('uint8')
        imsave(f'predicts/test/{id_}_proba.png', img)


if __name__ == '__main__':
    Fire(plot_all)
