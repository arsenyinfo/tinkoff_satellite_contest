import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, concatenate
from keras.layers.core import Activation
from keras.optimizers import SGD
from keras import backend as K
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from src.unet import double_conv_layer
from src.plot import PatchCombiner
from src.utils import logger
from src.aug import augment

np.random.seed(42)


def get_callbacks():
    model_checkpoint = ModelCheckpoint('models/height.h5',
                                       monitor='loss',
                                       save_best_only=True, verbose=0)
    es = EarlyStopping(monitor='loss', min_delta=0, patience=12, verbose=1, mode='auto')
    reducer = ReduceLROnPlateau(monitor='loss', min_lr=1e-6, verbose=1, factor=0.1, patience=3)
    return [model_checkpoint, es, reducer]


def parse_data():
    train = pd.read_csv('data/task2/csv/train.csv')
    train = train[train.img_name == 'swir']

    test = pd.read_csv('data/task2/csv/test.csv')
    test = test[test.img_name == 'swir']

    x_data = imread('data/task2/swir.tif')
    x, y, z = x_data.shape

    add_x_data = imread('data/task2/mul.tif')
    add_x_data = resize(add_x_data, x_data.shape)
    huge_x_data = imread('data/task2/pan.tif')
    huge_x_data = resize(huge_x_data, x_data.shape)

    x_data = np.dstack((x_data, add_x_data, huge_x_data))

    z = 1
    y_data = np.zeros((x, y, z))

    size = 6

    for _, row in train.iterrows():
        for a in range(size):
            for b in range(size):
                y_data[int(row.y + a - size / 2), int(row.x + b - size / 2), ...] = row.height / 3

    for _, row in test.iterrows():
        for a in range(size):
            for b in range(size):
                y_data[int(row.y + a - size / 2), int(row.x + b - size / 2), ...] = -1

    return x_data, y_data


def gen(x_data, y_data, batch_size=32, patch_size=128):
    shape = x_data.shape

    while True:
        patches, targets = [], []

        while len(patches) < batch_size:
            w = np.random.randint(0, shape[0] - patch_size)
            h = np.random.randint(0, shape[1] - patch_size)

            patch = x_data[w: w + patch_size, h: h + patch_size, ...]
            target = y_data[w: w + patch_size, h: h + patch_size, ...]

            if target.min() < 0:
                continue

            # print(patch.shape, target.shape, 'before aug')
            patch, target = augment(patch, ta
            rget, stretched_size=150)

            patches.append(patch)
            targets.append(target)

        yield np.array(patches).astype('float32'), np.array(targets).astype('uint8')


def zf_unet(channels, filters=8, dropout_val=0.2, batch_norm=True):
    inputs = Input((128, 128, channels))

    bn = BatchNormalization()(inputs)
    conv_224 = double_conv_layer(bn, filters, dropout_val, batch_norm)
    pool_112 = MaxPooling2D(pool_size=(2, 2))(conv_224)

    conv_112 = double_conv_layer(pool_112, 2 * filters, dropout_val, batch_norm)
    pool_56 = MaxPooling2D(pool_size=(2, 2))(conv_112)

    conv_56 = double_conv_layer(pool_56, 4 * filters, dropout_val, batch_norm)
    pool_28 = MaxPooling2D(pool_size=(2, 2))(conv_56)

    conv_28 = double_conv_layer(pool_28, 8 * filters, dropout_val, batch_norm)
    pool_14 = MaxPooling2D(pool_size=(2, 2))(conv_28)

    conv_14 = double_conv_layer(pool_14, 16 * filters, dropout_val, batch_norm)
    pool_7 = MaxPooling2D(pool_size=(2, 2))(conv_14)

    conv_7 = double_conv_layer(pool_7, 32 * filters, dropout_val, batch_norm)

    up_14 = concatenate([UpSampling2D(size=(2, 2))(conv_7), conv_14], axis=-1)
    up_conv_14 = double_conv_layer(up_14, 16 * filters, dropout_val, batch_norm)

    up_28 = concatenate([UpSampling2D(size=(2, 2))(up_conv_14), conv_28], axis=-1)
    up_conv_28 = double_conv_layer(up_28, 8 * filters, dropout_val, batch_norm)

    up_56 = concatenate([UpSampling2D(size=(2, 2))(up_conv_28), conv_56], axis=-1)
    up_conv_56 = double_conv_layer(up_56, 4 * filters, dropout_val, batch_norm)

    up_112 = concatenate([UpSampling2D(size=(2, 2))(up_conv_56), conv_112], axis=-1)
    up_conv_112 = double_conv_layer(up_112, 2 * filters, dropout_val, batch_norm)

    up_224 = concatenate([UpSampling2D(size=(2, 2))(up_conv_112), conv_224], axis=-1)
    up_conv_224 = double_conv_layer(up_224, filters, 0, batch_norm)

    conv_final = Conv2D(1, (1, 1))(up_conv_224)
    conv_final = Activation('relu')(conv_final)

    model = Model(inputs, conv_final)
    return model


def loss(y_true, y_pred):
    return K.mean(K.square((y_pred - y_true) * y_true), axis=-1)


def make_preds():
    x_data, y_data = parse_data()
    train_gen = gen(x_data, y_data)

    model = zf_unet(channels=x_data.shape[-1])
    model.compile(optimizer='adam',
                  loss=loss)

    model.fit_generator(train_gen,
                        steps_per_epoch=200,
                        epochs=20,
                        callbacks=get_callbacks(),
                        max_queue_size=10,
                        workers=1, )

    model.compile(optimizer=SGD(clipvalue=3, clipnorm=1., momentum=.9),
                  loss=loss)

    model.fit_generator(train_gen,
                        steps_per_epoch=200,
                        epochs=20,
                        callbacks=get_callbacks(),
                        max_queue_size=10,
                        workers=1, )

    combiner = PatchCombiner(target_shape=y_data.shape,
                             batch_size=32,
                             window=128,
                             overlay=8,
                             x_data=x_data,
                             predict=load_model('models/height.h5', compile=False).predict,
                             verbose=0)

    img = combiner.process(png_ready=False)
    return img * 3


def get_height(img, x, y):
    size = 4
    result = []
    for a in range(size):
        for b in range(size):
            result.append(img[int(x + a - size / 2), int(y + b - size / 2), ...])

    return np.mean(result)


def main():
    train = pd.read_csv('data/task2/csv/train.csv')
    train = train[train.img_name == 'swir']

    test = pd.read_csv('data/task2/csv/test.csv')
    test = test[test.img_name == 'swir']

    img = make_preds()

    train_height = []
    test_height = []

    for _, row in train.iterrows():
        id_ = row['id']
        x = row['y']
        y = row['x']

        train_height.append({'id': id_,
                             'pred': get_height(img, x, y)})

    for _, row in test.iterrows():
        id_ = row['id']
        x = row['y']
        y = row['x']

        test_height.append({'id': id_,
                            'pred': get_height(img, x, y)})

    df = pd.merge(train, pd.DataFrame(train_height), on='id', how='inner')
    assert len(df) == len(train)

    lr = LinearRegression()
    x_data = df['pred'].values.reshape(-1, 1)
    y_data = df['height'].values

    score = cross_val_score(lr, x_data, y_data, scoring='neg_mean_squared_error', cv=5, n_jobs=-1).mean()
    logger.info(f'Score is {score:.3f}')

    lr.fit(x_data, y_data)
    logger.info(f'Slope is {lr.coef_[0]}, intercept is {lr.intercept_}')

    test_df = pd.DataFrame(test_height)
    test_df['height'] = lr.predict(test_df['pred'].values.reshape(-1, 1))
    test_df = test_df[['id', 'height']]

    test_df['height'] = test_df['height'].apply(lambda x: int(round(x / 3)) * 3)
    test_df.to_csv('predicts/task2_result.csv', index=False)


if __name__ == '__main__':
    main()
