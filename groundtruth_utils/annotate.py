from matplotlib import pyplot as plt
from gluoncv import model_zoo, utils
from gluoncv.model_zoo.alpha_pose import get_alphapose
from gluoncv.data.transforms.pose import detector_to_alpha_pose, heatmap_to_coord_alpha_pose
import mxnet as mx
import numpy as np

from .detector import DetectorType, Detector
from .weights import download_weights, ALPHAPOSE_MXNET_WEIGHTS_ID


def annotate_image(image_path, detector_type=DetectorType.PYTORCH_YOLOV4):
    bbox_conf_threshold = 0.3
    detector = Detector(detector_type, conf_threshold=bbox_conf_threshold)

    model_zoo.get_model('alpha_pose_resnet101_v1b_coco', pretrained=True)
    alphapose_net = get_alphapose(
        name='resnet101_v1b', dataset='coco', pretrained=False,
        num_joints=33, norm_layer=mx.gluon.nn.BatchNorm,
        norm_kwargs={'use_global_stats': False})
    # alphapose_net.load_parameters(get_model_file(full_name, tag=pretrained, root=root))
    params = download_weights(ALPHAPOSE_MXNET_WEIGHTS_ID)
    if params is None:
        return None

    alphapose_net.load_parameters(params)

    img, class_ids, scores, bounding_boxes = detector.detect(image_path)

    # Use NDArray as long as mxnet's alphapose is used
    if isinstance(class_ids, np.ndarray):
        class_ids = mx.nd.array(class_ids)
    if isinstance(scores, np.ndarray):
        scores = mx.nd.array(scores)
    if isinstance(bounding_boxes, np.ndarray):
        bounding_boxes = mx.nd.array(bounding_boxes)

    img_orig_size = mx.image.imread(image_path).shape[:2]
    img_reduced_size = img.shape[:2]
    img_upscale = img_orig_size[0] / img_reduced_size[0]

    pose_input, upscale_bbox = detector_to_alpha_pose(
        img, class_ids, scores, bounding_boxes, thr=bbox_conf_threshold)
    predicted_heatmap = alphapose_net(pose_input)
    pred_coords, confidence = heatmap_to_coord_alpha_pose(predicted_heatmap, upscale_bbox)

    pred_coords = pred_coords[:, np.r_[0:17, 25]]  # 0-17 Coco keypoints, 25 = mpii Neck keypoint
    confidence = confidence[:, np.r_[0:17, 25]]

    # ax = utils.viz.plot_keypoints(img, pred_coords, confidence,
    #                               class_ids, bounding_boxes, scores,
    #                               box_thresh=bbox_conf_threshold, keypoint_thresh=0.2)
    # plt.show()

    boxes_upscaled = bounding_boxes[0] * img_upscale
    pred_coords_upscaled = pred_coords * img_upscale

    boxes_in_x_y_w_h = []
    for i, bbox in enumerate(boxes_upscaled.asnumpy().tolist()):
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        boxes_in_x_y_w_h.append([xmin, ymin, xmax - xmin, ymax - ymin])

    zipped = list(zip(boxes_in_x_y_w_h, pred_coords_upscaled.asnumpy().tolist()))
    return list(map(lambda r: {'bbox': r[0], 'keypoints': r[1]}, zipped))
