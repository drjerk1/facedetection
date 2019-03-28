from errors import *
from signals import *
from useful import cosine_dist_norm
import numpy as np
from sklearn.cluster import KMeans

min_feautures = 10
default_num_feautures = 30

def feauture_select(feauture_list, num_feautures=None):
    feauture_list = np.array(feauture_list)

    if num_feautures is None:
        num_feautures = default_num_feautures
    else:
        try:
            num_feautures = int(num_feautures)
        except (TypeError, ValueError):
            raise ErrorSignal(invalid_argument_value)

    if feauture_list.shape[0] < min_feautures or feauture_list.shape[0] < num_feautures:
        raise ErrorSignal(too_few_samples)

    np.random.shuffle(feauture_list)
    train = feauture_list[:len(feauture_list)//2]
    val = feauture_list[len(feauture_list)//2:]

    kmeans = KMeans(num_feautures)
    kmeans.fit(feauture_list)
    feautures = np.array(kmeans.cluster_centers_)
    threshold = np.max(cosine_dist_norm(feautures[kmeans.predict(val)], val))

    return feautures, threshold
