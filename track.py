import numpy as np

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort


class Tracking:
    def __init__(self):
        self.class_names = [
            "person",
            "bicycle",
            "car",
            "motorbike",
            "aeroplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "sofa",
            "pottedplant",
            "bed",
            "diningtable",
            "toilet",
            "tvmonitor",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3))

    def initialize_deepsort(self):
        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        deepsort = DeepSort(
            cfg.DEEPSORT.REID_CKPT,
            max_dist=cfg.DEEPSORT.MAX_DIST,
            # minimum confidence parameter sets the minimum tracking confidence required for an object detection to be considered in the tracking process
            min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            # max_age: if an object tracing ID is lost, this parameter determines how many frames the tracker should wait before assigning a new iD
            max_age=cfg.DEEPSORT.MAX_AGE,
            n_init=cfg.DEEPSORT.N_INIT,
            # nn_budget: It sets the budget for nearest-neighbor search
            nn_budget=cfg.DEEPSORT.NN_BUDGET,
            use_cuda=False,
        )
        return deepsort

    def draw_boxes(self):
        pass
