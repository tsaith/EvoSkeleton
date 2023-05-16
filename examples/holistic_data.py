import numpy as np


class HolisticData:

    def __init__(self):

        self.image_width = None
        self.image_height = None

        self.num_pose_landmarks = 33
        self.num_hand_landmarks = 21
        self.num_facemesh_landmarks = 468
        self.dims = 4
     
        self.bl_pose = None
        self.mp_results = None
        
        self.has_pose = True
        self.pose = np.zeros((self.num_pose_landmarks, self.dims), dtype=np.float32)
        self.left_hand = np.zeros((self.num_hand_landmarks, self.dims), dtype=np.float32)
        self.right_hand = np.zeros((self.num_hand_landmarks, self.dims), dtype=np.float32)
        self.facemesh = np.zeros((self.num_facemesh_landmarks, self.dims), dtype=np.float32)


        # Pose indexes
        self.index_nose = 0
        self.index_left_eye_inner = 1
        self.index_left_eye = 2
        self.index_left_eye_outer = 3
        self.index_right_eye_inner = 4
        self.index_right_eye = 5
        self.index_right_eye_outer = 6
        self.index_left_ear = 7
        self.index_right_ear = 8
        self.index_mouth_left = 9
        self.index_mouth_right = 10
        self.index_left_shoulder = 11
        self.index_right_shoulder = 12
        self.index_left_elbow = 13
        self.index_right_elbow = 14
        self.index_left_wrist = 15
        self.index_right_wrist = 16
        self.index_left_pinky = 17
        self.index_right_pinky = 18
        self.index_left_index = 19
        self.index_right_index = 20
        self.index_left_thumb = 21
        self.index_right_thumb = 22
        self.index_left_hip = 23
        self.index_right_hip = 24
        self.index_left_knee = 25
        self.index_right_knee = 26
        self.index_left_ankle = 27
        self.index_right_ankle = 28
        self.index_left_heel = 29
        self.index_right_heel = 30
        self.index_left_foot_index = 31
        self.index_right_foot_index = 32

    def set_image_size(self, width, height):
        
        self.image_width = width
        self.image_height = height

    def update(self, mp_results):

        self.mp_results = mp_results

        # Pose
        self.update_pose()


    def update_with_baseline(self, bl_pose):

        self.bl_pose = bl_pose

    def update_pose(self):

        pose_landmarks = self.mp_results.pose_landmarks

        for i in range(self.num_pose_landmarks):

            x = pose_landmarks.landmark[i].x           
            y = pose_landmarks.landmark[i].y
            z = pose_landmarks.landmark[i].z
            visibility = pose_landmarks.landmark[i].visibility          

            self.pose[i, 0] = x
            self.pose[i, 1] = y
            self.pose[i, 2] = z
            self.pose[i, 3] = visibility
    
    def get_pose(self):
        return self.pose

    def get_nose(self):
        return self.pose[self.index_nose]

    def get_left_eye_inner(self):
        return self.pose[self.index_left_eye_inner]

    def get_left_eye(self):
        return self.pose[self.index_left_eye]

    def get_left_eye_outer(self):
        return self.pose[self.index_left_eye_outer]

    def get_right_eye_inner(self):
        return self.pose[self.index_right_eye_inner]

    def get_right_eye(self):
        return self.pose[self.index_right_eye]

    def get_right_eye_outer(self):
        return self.pose[self.index_right_eye_outer]

    def get_left_ear(self):
        return self.pose[self.index_left_ear]

    def get_right_ear(self):
        return self.pose[self.index_right_ear]

    def get_mouth_left(self):
        return self.pose[self.index_mouth_left]

    def get_mouth_right(self):
        return self.pose[self.index_mouth_right]

    def get_left_shoulder(self):
        return self.pose[self.index_left_shoulder]

    def get_right_shoulder(self):
        return self.pose[self.index_right_shoulder]

    def get_left_elbow(self):
        return self.pose[self.index_left_elbow]

    def get_right_elbow(self):
        return self.pose[self.index_right_elbow]

    def get_left_wrist(self):
        return self.pose[self.index_left_wrist]

    def get_right_wrist(self):
        return self.pose[self.index_right_wrist]

    def get_left_pinky(self):
        return self.pose[self.index_left_pinky]

    def get_right_pinky(self):
        return self.pose[self.index_right_pinky]

    def get_left_index(self):
        return self.pose[self.index_left_index]

    def get_right_index(self):
        return self.pose[self.index_right_index]

    def get_left_thumb(self):
        return self.pose[self.index_left_thumb]

    def get_right_thumb(self):
        return self.pose[self.index_right_thumb]

    def get_left_hip(self):
        return self.pose[self.index_left_hip]

    def get_right_hip(self):
        return self.pose[self.index_right_hip]

    def get_left_knee(self):
        return self.pose[self.index_left_knee]

    def get_right_knee(self):
        return self.pose[self.index_right_knee]

    def get_left_ankle(self):
        return self.pose[self.index_left_ankle]

    def get_right_ankle(self):
        return self.pose[self.index_right_ankle]

    def get_left_heel(self):
        return self.pose[self.index_left_heel]

    def get_right_heel(self):
        return self.pose[self.index_right_heel]

    def get_left_foot_index(self):
        return self.pose[self.index_left_foot_index]

    def get_right_foot_index(self):
        return self.pose[self.index_right_foot_index]

    def to_pixel_point(self, point):

        point_pixel = np.zeros(3, dtype=np.float32) 

        point_pixel[0] = point[0] * self.image_width
        point_pixel[1] = point[1] * self.image_height
        point_pixel[2] = point[2] * self.image_width

        return point_pixel

    def get_pose_pixel(self):

        pose_pixel = np.zeros_like(self.pose)

        pose_pixel[:, 0] = self.pose[:, 0] * self.image_width
        pose_pixel[:, 1] = self.pose[:, 1] * self.image_height
        pose_pixel[:, 2] = self.pose[:, 2] * self.image_width

        return pose_pixel