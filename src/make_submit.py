from glob import glob

import numpy as np
import cv2
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import HuberRegressor, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from skimage.morphology import disk
from skimage.filters.rank import enhance_contrast
from tqdm import tqdm
from skimage.measure import find_contours
from sklearn.metrics import make_scorer

from src.utils import logger


def mape(y_true, y_pred):
    return -np.mean(np.abs((y_true - y_pred) / y_true))


def count_contours_cv(x):
    _, conts, h = cv2.findContours(np.copy(x), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return len(conts)


def make_img_features(id_, train=True):
    path = 'train' if train else 'test'
    pred = cv2.imread(f'predicts/{path}/{id_}_proba.png')[..., 0]

    pred_bin = np.copy(enhance_contrast(np.copy(pred), disk(3)))
    pred_bin[pred_bin >= 128] = 255
    pred_bin[pred_bin < 128] = 0

    features_pred = [len(find_contours(pred, .9)),
                     len(find_contours(pred_bin, .9)),
                     count_contours_cv(pred),
                     count_contours_cv(pred_bin),
                     pred.mean(),
                     pred.std(), ]

    features_pred = np.array(features_pred)
    return features_pred


def count_cars():
    gt = pd.read_csv('data/imgs.csv')
    worker = Parallel(n_jobs=-1, verbose=1, backend='threading')

    train_features = worker(delayed(make_img_features)(id_) for id_ in gt['id'].values)
    x_data = np.vstack(tuple(train_features))
    y_data = gt['car_count'].values

    x_test_ids = [x.split('/')[-1].split('.')[0] for x in glob('data/tif/tif_test/*.tif')]

    test_features = worker(delayed(make_img_features)(id_, train=False) for id_ in x_test_ids)
    x_test = np.vstack(tuple(test_features))

    scorer = make_scorer(mape, greater_is_better=True)
    scaler = StandardScaler()

    x_data = scaler.fit_transform(x_data)
    x_test = scaler.transform(x_test)

    preds = []
    for est in HuberRegressor(), BayesianRidge(), RandomForestRegressor():
        score = cross_val_score(est, x_data, y_data, scoring=scorer, cv=5, )
        logger.info(f'Score for {est.__class__} is {score.mean():.3f}}')

        est.fit(x_data, y_data)
        preds.append(est.predict(x_test))

    preds = np.array(preds).mean(axis=0)

    pd.DataFrame({'id': x_test_ids, 'car_count': [int(x) for x in preds]}) \
        .to_csv('predicts/final/imgs.csv', index=False)


def copy_imgs():
    x_test_ids = [x.split('/')[-1].split('.')[0] for x in glob('data/tif/tif_test/*.tif')]
    for id_ in tqdm(x_test_ids, desc='making discrete images'):
        img = cv2.imread(f'predicts/test/{id_}_proba.png')
        threshold = 128
        img[img >= threshold] = 255
        img[img < threshold] = 0
        cv2.imwrite(f'predicts/final/masks_test/{id_}.png', img)


if __name__ == '__main__':
    count_cars()
    copy_imgs()
