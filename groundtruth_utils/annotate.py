from gluoncv import model_zoo, data
from gluoncv.model_zoo.alpha_pose import get_alphapose
from gluoncv.data.transforms.pose import detector_to_alpha_pose, heatmap_to_coord_alpha_pose
import mxnet as mx
import numpy as np

from .log import logger


def annotate_image(image_path):
    detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
    model_zoo.get_model('alpha_pose_resnet101_v1b_coco', pretrained=True)
    bbox_threshold = 0.5

    alphapose_net = get_alphapose(
        name='resnet101_v1b', dataset='coco',
        num_joints=18, norm_layer=mx.gluon.nn.BatchNorm,
        norm_kwargs={'use_global_stats': False})

    detector.reset_class(["person"], reuse_weights=['person'])

    x, img = data.transforms.presets.yolo.load_test(image_path, short=512)

    class_ids, scores, bounding_boxes = detector(x)

    # Filter bounding box detections by threshold
    np_scores = scores.asnumpy()
    np_threshold_indices = np.where(np_scores[:, :, 0] > bbox_threshold)

    filtered_scores = scores[np_threshold_indices]
    filtered_boxes = bounding_boxes[np_threshold_indices]
    filtered_class_ids = class_ids[np_threshold_indices]

    pose_input, upscale_bbox = detector_to_alpha_pose(
        img, filtered_class_ids, filtered_scores, filtered_boxes, thr=bbox_threshold)
    predicted_heatmap = alphapose_net(pose_input)
    pred_coords, confidence = heatmap_to_coord_alpha_pose(predicted_heatmap, upscale_bbox)

    # return {
    #     'boxes': filtered_boxes.asnumpy().tolist(),
    #     #'yolo_boxes': upscale_bbox,
    #     'keypoints': pred_coords.asnumpy().tolist()
    # }

    zipped = list(zip(filtered_boxes.asnumpy().tolist(), pred_coords.asnumpy().tolist()))
    return list(map(lambda r: {'bbox': r[0], 'keypoints': r[1]}, zipped))
