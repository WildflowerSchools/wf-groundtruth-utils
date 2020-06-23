from gluoncv import model_zoo, data
from gluoncv.model_zoo.alpha_pose import get_alphapose
from gluoncv.data.transforms.pose import detector_to_alpha_pose, heatmap_to_coord_alpha_pose
import mxnet as mx
import numpy as np


def annotate_image(image_path):
    detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
    model_zoo.get_model('alpha_pose_resnet101_v1b_coco', pretrained=True)
    bbox_threshold = 0.3

    alphapose_net = get_alphapose(
        name='resnet101_v1b', dataset='coco', pretrained=False,
        num_joints=33, norm_layer=mx.gluon.nn.BatchNorm,
        norm_kwargs={'use_global_stats': False})
    # alphapose_net.load_parameters(get_model_file(full_name, tag=pretrained, root=root))
    alphapose_net.load_parameters("../ignore/models/duc_se.params")

    detector.reset_class(["person"], reuse_weights=['person'])

    x, img = data.transforms.presets.yolo.load_test(image_path, short=512)

    img_orig_size = mx.image.imread(image_path).shape[:2]
    img_reduced_size = img.shape[:2]
    img_upscale = img_orig_size[0] / img_reduced_size[0]

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

    pred_coords = pred_coords[:, np.r_[0:17, 25]]  # 0-17 Coco keypoints, 25 = mpii Neck keypoint
    confidence = confidence[:, np.r_[0:17, 25]]

    # ax = utils.viz.plot_keypoints(img, pred_coords, confidence,
    #                              class_ids, bounding_boxes, scores,
    #                              box_thresh=bbox_threshold, keypoint_thresh=0.2)
    # plt.show()

    # return {
    #     'boxes': filtered_boxes.asnumpy().tolist(),
    #     #'yolo_boxes': upscale_bbox,
    #     'keypoints': pred_coords.asnumpy().tolist()
    # }

    filtered_boxes_upscaled = filtered_boxes * img_upscale
    pred_coords_upscaled = pred_coords * img_upscale

    boxes_in_x_y_w_h = []
    for i, bbox in enumerate(filtered_boxes_upscaled.asnumpy().tolist()):
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        boxes_in_x_y_w_h.append([xmin, ymin, xmax - xmin, ymax - ymin])

    zipped = list(zip(boxes_in_x_y_w_h, pred_coords_upscaled.asnumpy().tolist()))
    return list(map(lambda r: {'bbox': r[0], 'keypoints': r[1]}, zipped))
