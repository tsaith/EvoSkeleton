import numpy as np


class BaselineData:

    def __init__(self):

        self.num_pose_landmarks = 17
        self.dims = 3
        self.dims_2d = 3
     
        self.has_pose = True
        self.pose = np.zeros((self.num_pose_landmarks, self.dims), dtype=np.float32)

        self.index_hip = 0
        self.index_right_hip = 1
        self.index_right_knee = 2
        self.index_right_ankle = 3
        self.index_left_hip = 4
        self.index_left_knee = 5
        self.index_left_ankle = 6
        self.index_spine02 = 7
        self.index_neck01 = 8
        self.index_nose = 9
        self.index_head_top = 10
        self.index_left_shoulder = 11
        self.index_left_elbow = 12
        self.index_left_wrist = 13
        self.index_right_shoulder = 14
        self.index_right_elbow = 15
        self.index_right_wrist = 16

        self.holistic_data = None 

        self.bl_pose = None
        self.bl_inputs = None

        self.stat_3d = None  

        use_dim = [ 
            3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 51,
            52, 53, 54, 55, 56, 57, 58, 59, 75, 76, 77, 78, 79, 80, 81, 82, 83
        ]
        self.use_dim = np.array(use_dim, dtype=np.int32)
        self.use_dim_with_hip = np.hstack((np.arange(3), self.use_dim)) # Add hip

        self.pose_means = np.zeros((self.num_pose_landmarks, self.dims), dtype=np.float32)
        self.pose_stddevs = np.zeros((self.num_pose_landmarks, self.dims), dtype=np.float32)

    def update_with_bl_pose_2d(self, pose):

        for i in range(self.num_pose_landmarks):
            self.pose[i, 0:2] = self.get_point_from_bl_pose_2d(i, pose)

    def update_with_bl_pose_3d(self, pose):

        for i in range(self.num_pose_landmarks):
            self.pose[i, :] = self.get_point_from_bl_pose_3d(i, pose)

    def update_with_pose(self, pose, is_3d=True):

        for i in range(self.num_pose_landmarks):

            if is_3d:
                self.pose[i, :] = pose[i, :]
            else:    
                self.pose[i, 0:2] = pose[i, 0:2]


    def update_with_holistic(self, holistic_data):

        self.holistic_data = holistic_data 
        data = holistic_data

        left_ear = self.get_point(data.get_left_ear())
        right_ear = self.get_point(data.get_right_ear())
        head = 0.5*(left_ear + right_ear)

        # Hip
        left_hip = self.get_point(data.get_left_hip())
        right_hip = self.get_point(data.get_right_hip())
        hip = 0.5*(left_hip + right_hip)

        self.pose[self.index_hip, :] = hip

        # Right hip
        self.pose[self.index_right_hip, :] = right_hip

        # Right knee
        right_knee = self.get_point(data.get_right_knee())

        self.pose[self.index_right_knee, :] = right_knee

        # Right ankle
        right_ankle = self.get_point(data.get_right_ankle())

        self.pose[self.index_right_ankle, :] = right_ankle

        # Left hip
        self.pose[self.index_left_hip, :] = left_hip

        # Left knee
        left_knee = self.get_point(data.get_left_knee())

        self.pose[self.index_left_knee, :] = left_knee

        # Left ankle
        left_ankle = self.get_point(data.get_left_ankle())

        self.pose[self.index_left_ankle, :] = left_ankle

        # Spine02
        left_shoulder = self.get_point(data.get_left_shoulder())
        right_shoulder = self.get_point(data.get_right_shoulder())
        shoulder_center = 0.5*(left_shoulder + right_shoulder)
        spine02 = 0.5*(hip + shoulder_center)

        self.pose[self.index_spine02, :] = spine02

        # Neck01
        neck01 = 0.5*(head + shoulder_center)
        self.pose[self.index_neck01, :] = neck01

        # Nose
        nose = self.get_point(data.get_nose())
        self.pose[self.index_nose, :] = nose

        # Head top
        head_top = head 
        self.pose[self.index_head_top, :] = head_top

        # Left shoulder
        left_shoulder = self.get_point(data.get_left_shoulder())
        self.pose[self.index_left_shoulder, :] = left_shoulder

        # Left elbow
        left_elbow = self.get_point(data.get_left_elbow())
        self.pose[self.index_left_elbow, :] = left_elbow

        # Left wrist
        left_wrist = self.get_point(data.get_left_wrist())
        self.pose[self.index_left_wrist, :] = left_wrist

        # right shoulder
        right_shoulder = self.get_point(data.get_right_shoulder())
        self.pose[self.index_right_shoulder, :] = right_shoulder

        # right elbow
        right_elbow = self.get_point(data.get_right_elbow())
        self.pose[self.index_right_elbow, :] = right_elbow

        # right wrist
        right_wrist = self.get_point(data.get_right_wrist())
        self.pose[self.index_right_wrist, :] = right_wrist

        # Mapping Mediapipe pose to H3.6m camera space
        #self.mp_to_h36m_camera_space()


    def get_keypoint(self, index):
        return self.pose[index, :].copy()

    def get_point_from_bl_pose_2d(self, index, pose):

        x = pose[2*index]
        y = pose[2*index+1]

        point = np.array([x, y], dtype=np.float32)

        return point

    def get_point_from_bl_pose_3d(self, index, pose):

        x = pose[3*index]
        y = pose[3*index+1]
        z = pose[3*index+2]

        point = np.array([x, y, z], dtype=np.float32)

        return point

    def get_point(self, data):

        point = np.zeros(self.dims, dtype=np.float32)
        point[:] = data[0:self.dims]

        return point
    
    def get_pose_flattened(self):
        return self.pose.flatten() 
    
    def get_connect_pairs(self):

        hip = self.get_keypoint(self.index_hip)
        right_hip = self.get_keypoint(self.index_right_hip)
        right_knee = self.get_keypoint(self.index_right_knee)
        right_ankle = self.get_keypoint(self.index_right_ankle)
        left_hip = self.get_keypoint(self.index_left_hip)
        left_knee = self.get_keypoint(self.index_left_knee)
        left_ankle = self.get_keypoint(self.index_left_ankle)
        spine02 = self.get_keypoint(self.index_spine02)
        neck01 = self.get_keypoint(self.index_neck01)
        nose = self.get_keypoint(self.index_nose)
        head_top = self.get_keypoint(self.index_head_top)
        left_shoulder = self.get_keypoint(self.index_left_shoulder)
        left_elbow = self.get_keypoint(self.index_left_elbow)
        left_wrist = self.get_keypoint(self.index_left_wrist)
        right_shoulder = self.get_keypoint(self.index_right_shoulder)
        right_elbow = self.get_keypoint(self.index_right_elbow)
        right_wrist = self.get_keypoint(self.index_right_wrist)
    
        hip_spine02 = [hip, spine02]
        spine02_neck01 = [spine02, neck01]
        neck01_nose = [neck01, nose]
        nose_head_top = [nose, head_top]
        neck01_left_shoulder = [neck01, left_shoulder]
        left_shoulder_left_elbow = [left_shoulder, left_elbow]
        left_elbow_left_wrist = [left_elbow, left_wrist]
        neck01_right_shoulder = [neck01, right_shoulder]
        right_shoulder_right_elbow = [right_shoulder, right_elbow]
        right_elbow_right_wrist = [right_elbow, right_wrist]
        hip_left_hip = [hip, left_hip]
        left_hip_left_knee = [left_hip, left_knee]
        left_knee_left_ankle = [left_knee, left_ankle]
        hip_right_hip = [hip, right_hip]
        right_hip_right_knee = [right_hip, right_knee]
        right_knee_right_ankle = [right_knee, right_ankle]

        pairs = [
            hip_spine02, spine02_neck01, neck01_nose, nose_head_top,
            neck01_left_shoulder, left_shoulder_left_elbow, left_elbow_left_wrist,
            neck01_right_shoulder, right_shoulder_right_elbow, right_elbow_right_wrist,
            hip_left_hip, left_hip_left_knee, left_knee_left_ankle,
            hip_right_hip, right_hip_right_knee, right_knee_right_ankle
        ]

        return pairs

    def get_bl_inputs(self):

        num_points = self.num_pose_landmarks - 1
        self.bl_inputs = np.zeros([1, num_points*2], dtype=np.float32)

        pose = self.normalize()

        for i in range(num_points):

            j = i+1
            x = pose[j, 0]
            y = pose[j, 1]    

            self.bl_inputs[0, 2*i] = x
            self.bl_inputs[0, 2*i+1] = y

        return self.bl_inputs

    def to_pose_2d(self, pose):
        return pose.reshape((-1, 2)).copy()

    def normalize_value(self, value, mean, stddev):
        return (value - mean) /stddev

    def unnormalize_value(self, value, mean, stddev):
        return value*stddev + mean

    def add_hip(self, targets, is_3d=True):

        outputs = []
        dims = 3
        for target in targets:

            dims = 3 if is_3d else 2
            out = np.concatenate([np.zeros(dims), target], axis=0)
            outputs.append(out)

        return outputs

    def normalize(self, pose=None, data_means=None, data_stddevs=None):

        if pose is None:
            pose = self.pose

        if data_means is None:
            data_means = self.pose_means 

        if data_stddevs is None:
            data_stddevs = self.pose_stddevs

        output = np.zeros_like(self.pose)

        for i in range(self.num_pose_landmarks):

            if i == 0: # Hip
                output[i, :] = 0.0        
                continue

            for j in range(self.dims):

                mean = data_means[i, j]
                stddev = data_stddevs[i, j]
                value = pose[i, j]
                output[i, j] = self.normalize_value(value, mean, stddev)

        return output

    def unnormalize(self, pose=None, data_means=None, data_stddevs=None, is_3d=True):

        if pose is None:
            pose = self.pose

        if data_means is None:
            data_means = self.pose_means 

        if data_stddevs is None:
            data_stddevs = self.pose_stddevs

        output = np.zeros_like(self.pose)

        for i in range(self.num_pose_landmarks):

            if i == 0: # Hip
                output[i, :] = 0.0        
                continue

            for j in range(self.dims):

                mean = data_means[i, j]
                stddev = data_stddevs[i, j]
                value = pose[i, j]
                output[i, j] = self.unnormalize_value(value, mean, stddev)

        if not is_3d:
            output = output[:, 0:2]  

        return output

    def get_2d_dims_from_3d(self, dims_3d):

        out = []
        for i, e in enumerate(dims_3d):
            if i % 3 == 2:
                continue

            out.append(e)

        out = np.array(out, np.int32)

        return out
    
    def get_use_dim(self):
        return self.use_dim_with_hip

    def get_use_dim_inputs(self):
        return self.get_2d_dims_from_3d(self.use_dim)

    def get_use_dim_2d(self):
        return self.get_2d_dims_from_3d(self.use_dim_3d)

    def get_use_dim_3d(self):
        return self.use_dim_3d

    def set_stat_3d(self, stat_3d):

        self.stat_3d = stat_3d

        data_means = self.stat_3d['mean'][self.get_use_dim()]
        data_stddevs = self.stat_3d['std'][self.get_use_dim()]

        for i in range(self.num_pose_landmarks):

            self.pose_means[i, 0] = data_means[3*i]
            self.pose_means[i, 1] = data_means[3*i+1]
            self.pose_means[i, 2] = data_means[3*i+2]

            self.pose_stddevs[i, 0] = data_stddevs[3*i]
            self.pose_stddevs[i, 1] = data_stddevs[3*i+1]
            self.pose_stddevs[i, 2] = data_stddevs[3*i+2]


    def set_pose(self, pose):
        self.pose = pose

    def get_pose(self):
        return self.pose
    
    def get_pose_means(self):
        return self.pose_means

    def get_pose_stddevs(self):
        return self.pose_stddevs

    def to_mp_pose(self, width=640, height=480):

        pose = np.zeros_like(self.pose)
        for i in range(self.num_pose_landmarks):

            pose[i, 0] = self.pose[i, 0] * width
            pose[i, 1] = self.pose[i, 1] * height
            pose[i, 2] = self.pose[i, 2] * width * 0.5

        return pose

    def to_bl_pose(self, width=1000, height=1000):

        pose = np.zeros_like(self.pose)
        for i in range(self.num_pose_landmarks):

            pose[i, 0] = self.pose[i, 0] * width
            pose[i, 1] = self.pose[i, 1] * height
            pose[i, 2] = self.pose[i, 2] * width * 0.5

        return pose
