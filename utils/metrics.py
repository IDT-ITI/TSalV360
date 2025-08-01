from functools import partial
import numpy as np
from numpy import random

from skimage import exposure
from skimage import img_as_float
from skimage.transform import resize
import math
import cv2

EPSILON = np.finfo('float').eps


def inner_worker(saliency_map, gt_s):
    # mground_truth = mground_truth.astype(np.float32)
    # saliency_map = saliency_map.astype(np.float32)
    gt_sal = gt_s
    # Calculate metrics

    kl = KLD(saliency_map, gt_sal)
    kl2 = KLD(gt_sal, saliency_map)
    cc = CC(saliency_map, gt_sal)
    sim = SIM(saliency_map, gt_sal)


    return kl,cc,sim, kl2


def norm(s):
    return (s - s.min()) / (s.max() - s.min())



def normalize(x, method='standard', axis=None):
    x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res

def KLD(p, q):
    p = normalize(p, method='sum')
    q = normalize(q, method='sum')
    return np.sum(np.where(p != 0, p * np.log((p + EPSILON) / (q + EPSILON)), 0))




def CC(saliency_map1, saliency_map2):
    map1 = np.array(saliency_map1, copy=False)
    map2 = np.array(saliency_map2, copy=False)
    if map1.shape != map2.shape:
        map1 = resize(map1, map2.shape, order=3,
                      mode='constant')  # bi-cubic/nearest is what Matlab imresize() does by default
    # Normalize the two maps to have zero mean and unit std
    map1 = normalize(map1, method='standard')
    map2 = normalize(map2, method='standard')
    # Compute correlation coefficient
    return np.corrcoef(map1.ravel(), map2.ravel())[0, 1]


def SIM(saliency_map1, saliency_map2):
    map1 = np.array(saliency_map1, copy=False)
    map2 = np.array(saliency_map2, copy=False)
    if map1.shape != map2.shape:
        map1 = resize(map1, map2.shape, order=3,
                      mode='constant')  # bi-cubic/nearest is what Matlab imresize() does by default
    # Normalize the two maps to have values between [0,1] and sum up to 1
    map1 = normalize(map1, method='range')
    map2 = normalize(map2, method='range')
    map1 = normalize(map1, method='sum')
    map2 = normalize(map2, method='sum')
    # Compute histogram intersection
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)

