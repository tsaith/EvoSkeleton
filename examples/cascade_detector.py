
import numpy as np
import torch

from skeleton_utils import re_order_indices, estimate_stats, normalize, unNormalizeData
from skeleton_utils import convert_to_skeleton_3d_h36m17p, scale_skeleton_3d_from_h36m
import libs.model.model as libm


def get_pred(cascade, data):
    """
    Get prediction from a cascaded model
    """

    # forward pass to get prediction for the first stage
    num_stages = len(cascade)
    # for legacy code that does not have the num_blocks attribute
    for i in range(len(cascade)):
        cascade[i].num_blocks = len(cascade[i].res_blocks)
    prediction = cascade[0](data)
    # prediction for later stages
    for stage_idx in range(1, num_stages):
        prediction += cascade[stage_idx](data)

    return prediction


class CascadeDetector:

    def __init__(self):

        self.cascade = None
        self.stats = None

        self.mean_3d = None
        self.stddev_3d = None


    def load_model(self, model_path):

        ckpt = torch.load(model_path)

        # initialize the model
        self.cascade = libm.get_cascade()
        input_size = 32
        output_size = 48
        for stage_id in range(2):
            # initialize a single deep learner
            stage_model = libm.get_model(stage_id + 1,
                                         refine_3d=False,
                                         norm_twoD=False, 
                                         num_blocks=2,
                                         input_size=input_size,
                                         output_size=output_size,
                                         linear_size=1024,
                                         dropout=0.5,
                                         leaky=False)
            self.cascade.append(stage_model)
        
        self.cascade.load_state_dict(ckpt)
        self.cascade.eval()

    def load_stats(self, stats_path):
        self.stats = np.load(stats_path, allow_pickle=True).item()

    def predict(self, skeleton_2d):

        #skel_stats = estimate_stats(skeleton_2d, re_order_indices)
        norm_ske_gt = normalize(skeleton_2d, re_order_indices).reshape(1,-1)
        #norm_ske_gt = normalize(skeleton_2d, re_order_indices, stats=skel_stats).reshape(1,-1)
        pred = get_pred(self.cascade, torch.from_numpy(norm_ske_gt.astype(np.float32)))      

        #pred = pred.data.numpy()

        #print(f"dim_ignore_3d: {self.stats['dim_ignore_3d']}")      

        pred = unNormalizeData(pred.data.numpy(),
            self.stats['mean_3d'],
            self.stats['std_3d'],
            self.stats['dim_ignore_3d'])      
        print(f"ori pred shape: {pred.shape}")      
    
        return pred
