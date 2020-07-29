from enum import IntEnum, auto

from gluoncv import model_zoo, data
import mxnet as mx
import numpy as np
from yolov4.tool.class_names import COCO_NAMES
from yolov4.tool.config import YOLO_V4
from yolov4.tool.darknet2pytorch import Darknet
from yolov4.tool.torch_utils import do_detect, time
from yolov4.tool.utils import load_class_names
from yolov4.tool.weights import download_weights


class DetectorType(IntEnum):
    MXNET_YOLOV3 = auto()
    PYTORCH_YOLOV4 = auto()


class Detector(object):
    def __init__(self, detector_type=DetectorType.PYTORCH_YOLOV4, conf_threshold=0.5, use_cuda=False):
        self.detector_type = detector_type
        self.conf_threshold = conf_threshold
        self.use_cuda = use_cuda

        self._init_detector()

    def _init_detector(self):
        detector = None
        if self.detector_type == DetectorType.PYTORCH_YOLOV4:
            weight_file = download_weights()

            detector = Darknet(YOLO_V4)
            detector.load_weights(weight_file)

            if self.use_cuda:
                detector.cuda()

        elif self.detector_type == DetectorType.MXNET_YOLOV3:
            detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
            detector.reset_class(["person"], reuse_weights=['person'])

        self.detector = detector

    def _yolov4_detect(self, image_path):
        import cv2

        img = cv2.imread(image_path)
        resized_img = cv2.resize(img, (self.detector.width, self.detector.height))
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

        start = time.time()
        detections = np.array(do_detect(self.detector, resized_img, 0.05, 0.6, self.use_cuda))
        finish = time.time()

        class_names = load_class_names(COCO_NAMES)
        person_class_id = class_names.index('person')

        filtered_class_ids = np.array(detections)[0, :, 6] == person_class_id
        filtered_detections = np.array(detections)[:, filtered_class_ids, :]

        boxes = filtered_detections[:, :, 0:4]

        # Convert from 0.0-1.0 float value to specific pixel location
        width = resized_img.shape[1]
        height = resized_img.shape[0]
        # xmin = (boxes[:, :, 0] - (boxes[:, :, 2] / 2.0)) * width
        # ymin = (boxes[:, :, 1] - (boxes[:, :, 3] / 2.0)) * height
        # xmax = (boxes[:, :, 0] + (boxes[:, :, 2] / 2.0)) * width
        # ymax = (boxes[:, :, 1] + (boxes[:, :, 3] / 2.0)) * height
        xmin = boxes[:, :, 0] * width
        ymin = boxes[:, :, 1] * height
        xmax = boxes[:, :, 2] * width
        ymax = boxes[:, :, 3] * height

        boxes[:, :, 0] = xmin
        boxes[:, :, 1] = ymin
        boxes[:, :, 2] = xmax
        boxes[:, :, 3] = ymax

        scores = filtered_detections[:, :, 5:6]
        class_ids = filtered_detections[:, :, 6:7]

        return resized_img, class_ids, scores, boxes

    def detect(self, image_path):
        if self.detector_type == DetectorType.PYTORCH_YOLOV4:
            img, class_ids, scores, bounding_boxes = self._yolov4_detect(image_path)
        elif self.detector_type == DetectorType.MXNET_YOLOV3:
            x, img = data.transforms.presets.yolo.load_test(image_path, short=512)
            class_ids, scores, bounding_boxes = self.detector(x)

        # Filter bounding box detections by threshold
        if isinstance(scores, mx.nd.NDArray):
            scores = scores.asnumpy()
        threshold_indices = np.where(scores[:, :, 0] > self.conf_threshold)

        if len(threshold_indices[0]) == 0:
            return img, [], [], []

        # Filter and retain the original array size
        filtered_scores = scores[threshold_indices][np.newaxis, ...]
        filtered_boxes = bounding_boxes[threshold_indices][np.newaxis, ...]
        filtered_class_ids = class_ids[threshold_indices][np.newaxis, ...]
        return img, filtered_class_ids, filtered_scores, filtered_boxes
