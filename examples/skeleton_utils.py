import numpy as np

pose_connection = [[0,1], [1,2], [2,3], [0,4], [4,5], [5,6], [0,7], [7,8],
                   [8,9], [9,10], [8,11], [11,12], [12,13], [8, 14], [14, 15], [15,16]]

# 16 out of 17 key-points are used as inputs in this examplar model
re_order_indices= [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16] # Without Neck


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

    print(f"orig_data shape: {orig_data.shape}")

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

def convert_holistic_to_skeleton_2d(data):

    num_points = 17
    dims = 2
    skel = np.zeros((num_points, dims), dtype=np.float32)

    def get_2d(p):
        return p[0:2]

    left_ear = get_2d(data.get_left_ear())
    right_ear = get_2d(data.get_right_ear())
    head = 0.5*(left_ear + right_ear)

    left_hip = get_2d(data.get_left_hip())
    right_hip = get_2d(data.get_right_hip())
    hip_center = 0.5*(left_hip + right_hip)

    left_shoulder = get_2d(data.get_left_shoulder())
    right_shoulder = get_2d(data.get_right_shoulder())
    shoulder_center = 0.5*(left_shoulder + right_shoulder)

    # Pelvis
    pelvis = hip_center
    #pelvis = 0.8*hip_center + 0.2*shoulder_center
    skel[0, :] = pelvis

    # Right hip
    skel[1, :] = right_hip

    # Right knee
    right_knee = get_2d(data.get_right_knee())

    skel[2, :] = right_knee

    # Right ankle
    right_ankle = get_2d(data.get_right_ankle())

    skel[3, :] = right_ankle

    # Left hip
    skel[4, :] = left_hip

    # Left knee
    left_knee = get_2d(data.get_left_knee())

    skel[5, :] = left_knee

    # Left ankle
    left_ankle = get_2d(data.get_left_ankle())

    skel[6, :] = left_ankle

    # Spine
    spine = 0.5*(hip_center + shoulder_center)

    skel[7, :] = spine

    # Thorax
    thorax = shoulder_center
    skel[8, :] = thorax

    # Neck
    neck = 0.7*shoulder_center + 0.3*head
    skel[9, :] = neck

    # Head top
    head_top = head 
    skel[10, :] = head_top

    # Left shoulder
    left_shoulder = get_2d(data.get_left_shoulder())
    skel[11, :] = left_shoulder

    # Left elbow
    left_elbow = get_2d(data.get_left_elbow())
    skel[12, :] = left_elbow

    # Left wrist
    left_wrist = get_2d(data.get_left_wrist())
    skel[13, :] = left_wrist

    # right shoulder
    right_shoulder = get_2d(data.get_right_shoulder())
    skel[14, :] = right_shoulder

    # right elbow
    right_elbow = get_2d(data.get_right_elbow())
    skel[15, :] = right_elbow

    # right wrist
    right_wrist = get_2d(data.get_right_wrist())
    skel[16, :] = right_wrist

    # Convert to pixel space
    skel[:, 0] = skel[:, 0] * data.image_width
    skel[:, 1] = skel[:, 1] * data.image_height

    return skel

def get_keypoint_indexes_h36m17p():

    indexes = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

    return indexes

def convert_to_skeleton_3d_h36m17p(data):

    num_points = 17
    dims = 3
    skel = np.zeros((num_points, dims), dtype=np.float32)

    indexes = get_keypoint_indexes_h36m17p()
    for i in range(num_points):

        j = indexes[i]
        skel[i, :] = data[j, :] 

    return skel

def scale_skeleton_3d_from_h36m(skel_3d, target_width, target_height):

    skel = skel_3d.copy()

    h36m_width = 1000.0
    h36m_height = 1000.0

    ratio_x = 1.0*target_width/h36m_width
    ratio_y = 1.0*target_height/h36m_height
    ratio_z = ratio_x

    skel[:, 0] = skel[:, 0] * ratio_x
    skel[:, 1] = skel[:, 1] * ratio_y
    skel[:, 2] = skel[:, 2] * ratio_z

    return skel