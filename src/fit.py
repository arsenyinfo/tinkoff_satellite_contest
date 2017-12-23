from functools import reduce

import numpy as np
from skimage.io import imread
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD
from fire import Fire

from src.unet import zf_unet
from src.loss import iou
from src.utils import FOLDS
from src.aug import augment


def get_callbacks(model, fold):
    model_checkpoint = ModelCheckpoint(f'models/{model}_{fold}.h5',
                                       monitor='val_loss',
                                       save_best_only=True, verbose=0)
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=12, verbose=1, mode='auto')
    reducer = ReduceLROnPlateau(monitor='val_loss', min_lr=1e-6, verbose=1, factor=0.1, patience=5)
    return [model_checkpoint, es, reducer]


def parse_file(file_id, is_train=True):
    x_path = 'tif_train' if is_train else 'tif_test'
    x_data = imread(f'data/tif/{x_path}/{file_id}.tif')
    x_data = np.expand_dims(x_data, -1)
    x_data = x_data.astype('float32') / 2047.
    assert x_data.shape[-1] == 1
    assert x_data.max() <= 1
    assert x_data.min() >= 0

    if not is_train:
        return x_data

    y_data = imread(f'data/masks/masks_train/{file_id}.png')
    y_data = y_data[..., 1:]
    y_data = y_data.astype('float32') / 255.
    y_data = y_data.astype('uint8')

    assert y_data.shape[-1] == 2
    assert set(y_data.flatten()) == {0, 1}
    assert 0 < y_data[..., 0].mean() < 1
    assert 0 < y_data[..., 1].mean() < 1

    return x_data, y_data


def get_data(batch_size, n_fold):
    folds = FOLDS

    train_ids = reduce(lambda x, y: x + y, [folds[i] for i in range(len(folds)) if i != n_fold])
    val_ids = folds[n_fold]

    x_train, y_train = zip(*(parse_file(file_id) for file_id in train_ids))
    x_val, y_val = zip(*(parse_file(file_id) for file_id in val_ids))

    return gen(x_train, y_train, batch_size=batch_size), gen(x_val, y_val, batch_size=batch_size)


def gen(x_data, y_data, batch_size=8, patch_size=256, stretched_size=300):
    shape = x_data[0].shape

    while True:
        patches, targets = [], []

        while len(patches) < batch_size:
            idx = np.random.randint(0, len(x_data))
            w = np.random.randint(0, shape[0] - patch_size)
            h = np.random.randint(0, shape[1] - patch_size)

            patch = x_data[idx][w: w + patch_size, h: h + patch_size, ...]
            target = y_data[idx][w: w + patch_size, h: h + patch_size, ...]

            patch, target = augment(patch, target, stretched_size=stretched_size)
            patches.append(patch)
            targets.append(target)

        yield np.array(patches).astype('float32'), np.array(targets).astype('uint8')


def main(name='unet', batch_size=32):
    for fold in range(5):
        model = zf_unet(channels=1, filters=8)
        model.compile(optimizer=SGD(momentum=.9),
                      loss='binary_crossentropy',
                      metrics=['accuracy', iou])

        train_gen, val_gen = get_data(batch_size=batch_size, n_fold=fold)

        model.fit_generator(train_gen,
                            steps_per_epoch=200,
                            epochs=1000,
                            validation_data=val_gen,
                            validation_steps=20,
                            callbacks=get_callbacks(name, fold=fold),
                            max_queue_size=10,
                            workers=1, )


if __name__ == '__main__':
    Fire(main)
