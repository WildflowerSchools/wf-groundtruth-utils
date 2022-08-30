import os


def detector_checkpoint_url():
    return os.getenv("DETECTOR_CHECKPOINT_URL", "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth")


def pose_checkpoint_url():
    return os.getenv("POSE_CHECKPOINT_URL", "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288-314c8528_20200708.pth")

