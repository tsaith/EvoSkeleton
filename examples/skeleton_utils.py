import numpy as np

pose_connection = [[0,1], [1,2], [2,3], [0,4], [4,5], [5,6], [0,7], [7,8],
                   [8,9], [9,10], [8,11], [11,12], [12,13], [8, 14], [14, 15], [15,16]]
# 16 out of 17 key-points are used as inputs in this examplar model
re_order_indices= [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16] # Skip nose


def re_order(skeleton):
    skeleton = skeleton.copy().reshape(-1,3)
    # permute the order of x,y,z axis
    skeleton[:,[0,1,2]] = skeleton[:, [0,2,1]]    
    return skeleton.reshape(96)


def estimate_stats(skeleton, re_order=None):

    skel = skeleton.copy()
    if re_order is not None:
        skel = skel[re_order].reshape(32)

    skel = skel.reshape(16, 2)
    mean_x = np.mean(skel[:,0])
    std_x = np.std(skel[:,0])
    mean_y = np.mean(skel[:,1])
    std_y = np.std(skel[:,1])
    std = (0.5*(std_x + std_y))

    stats = {'mean_x': mean_x, 'mean_y': mean_y, 'std': std}

    return stats

'''
def normalize(skeleton, re_order=None, stats=None):

    norm_skel = skeleton.copy()
    if re_order is not None:
        norm_skel = norm_skel[re_order].reshape(32)

    mean_x = stats['mean_x']
    mean_y = stats['mean_y']
    std = stats['std']

    norm_skel = norm_skel.reshape(16, 2)
    norm_skel[:,0] = (norm_skel[:,0] - mean_x)/std
    norm_skel[:,1] = (norm_skel[:,1] - mean_y)/std
    norm_skel = norm_skel.reshape(32)         

    return norm_skel
'''

def normalize(skeleton, re_order=None):

    norm_skel = skeleton.copy()
    if re_order is not None:
        norm_skel = norm_skel[re_order].reshape(32)

    norm_skel = norm_skel.reshape(16, 2)
    mean_x = np.mean(norm_skel[:,0])
    std_x = np.std(norm_skel[:,0])
    mean_y = np.mean(norm_skel[:,1])
    std_y = np.std(norm_skel[:,1])
    denominator = (0.5*(std_x + std_y))
    norm_skel[:,0] = (norm_skel[:,0] - mean_x)/denominator
    norm_skel[:,1] = (norm_skel[:,1] - mean_y)/denominator
    norm_skel = norm_skel.reshape(32)         

    return norm_skel

'''
def unnormalize(skeleton, stats=None):

    skel = skeleton.copy()

    mean_x = stats['mean_x']
    mean_y = stats['mean_y']
    std = stats['std']

    skel = skel.reshape(16, 3)

    skel[:, 0] = skel[:, 0]*std + mean_x
    skel[:, 1] = skel[:, 1]*std + mean_y

    skel = skel.reshape(48)         

    return skel
'''

def unNormalizeData(normalized_data, data_mean, data_std, dimensions_to_ignore):
    """
    Un-normalizes a matrix whose mean has been substracted and that has been 
    divided by standard deviation. Some dimensions might also be missing.
    
    Args
    normalized_data: nxd matrix to unnormalize
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dimensions_to_ignore: list of dimensions that were removed from the original data
    Returns
    orig_data: the unnormalized data
    """
    T = normalized_data.shape[0] # batch size
    D = data_mean.shape[0] # dimensionality
    orig_data = np.zeros((T, D), dtype=np.float32)
    dimensions_to_use = np.array([dim for dim in range(D)
                                    if dim not in dimensions_to_ignore])
    orig_data[:, dimensions_to_use] = normalized_data
    # multiply times stdev and add the mean
    stdMat = data_std.reshape((1, D))
    stdMat = np.repeat(stdMat, T, axis=0)
    meanMat = data_mean.reshape((1, D))
    meanMat = np.repeat(meanMat, T, axis=0)
    orig_data = np.multiply(orig_data, stdMat) + meanMat

    return orig_data

