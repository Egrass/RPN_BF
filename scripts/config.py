import math
""""Config about parametres used in train or test"""

class Config:
    def __init__(self):
        # GPU
        self.gpu_nums = 4

        # Anchor
        # anchor box scales
        self.anchor_box_scales = [40, 52, 68, 88, 114, 149, 193, 251, 326]
        # anchor box ratios
        self.anchor_box_ratios = [[0.41, 1]]
        # feature maps where anchors are based on
        self.feature_map = ['conv5_3']
        # encode
        self.ignore_threshold = 0.5
        self.prior_scaling = [1, 1, 1, 1]

        # Overlap
        # overlap for judge ground truth
        self.gt_p = 0.5
        self.gt_ng = 0.5
        # overlap for nms
        self.nms_overlap = 0.5

        # Image Preprocess
        # size to resize the smallest side of the image
        self.resize_size = [720, 960]
        self.origin_size = [480, 640]
        # Train
        # batch size
        self.batch_size = 1
        # weight decay
        self.weight_decay = 0.0005
        # max epoch
        self.max_steps = 80000
        # number of the classes
        self.num_classes = 2
        # learning rate in train
        self.learning_rate = [[0, 0.001], [60000, 0.0001]] # it means during the 0~60k mini-batches, learning rate is 0.001 and during the last 20k min-batches, learning rate is 0.0001
        # momentum
        self.momentum = 0.9

        # Image numbers
        self.test_images = 4025
        self.train_images = 1440

