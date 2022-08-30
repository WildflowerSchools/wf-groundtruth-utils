import os

import mmdet
import mmpose
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results)
from mmpose.datasets import DatasetInfo
import torch

from . import config
from .log import logger


class Annotate(object):
    def __init__(self, bbox_conf_threshold=0.3):
        self.bbox_conf_threshold = bbox_conf_threshold

        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda:0'

        self.det_model = init_detector(
            os.path.join(mmdet.__path__[0], ".mim/configs/yolox/yolox_x_8x8_300e_coco.py"),
            config.detector_checkpoint_url(),
            device=device)
        # build the pose model from a config file and a checkpoint file
        self.pose_model = init_pose_model(
            os.path.join(mmpose.__path__[0], ".mim/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_384x288.py"),
            config.pose_checkpoint_url(),
            device=device)

        self.dataset = self.pose_model.cfg.data['test']['type']
        self.dataset_info = self.pose_model.cfg.data['test'].get('dataset_info', None)
        if self.dataset_info is None:
            logger.error(
                'Please set `dataset_info` in the config.'
                'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                DeprecationWarning)
            raise Exception
        else:
            self.dataset_info = DatasetInfo(self.dataset_info)

    def annotate_image(self, image_path):
        mmdet_results = inference_detector(self.det_model, image_path)
        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, 1) # 1 = PERSON

        pose_results, returned_outputs = inference_top_down_pose_model(
            self.pose_model,
            image_path,
            person_results,
            bbox_thr=self.bbox_conf_threshold,
            format='xyxy',
            dataset=self.dataset,
            dataset_info=self.dataset_info,
            return_heatmap=False,
            outputs=None)

        pose_results = list(map(lambda r: {
            'bbox': [
                float(r['bbox'][0]),
                float(r['bbox'][1]),
                float(r['bbox'][2] - r['bbox'][0]),
                float(r['bbox'][3] - r['bbox'][1]),
                float(r['bbox'][4])
            ],
            'keypoints': r['keypoints'].tolist()
        }, pose_results))
        return pose_results
