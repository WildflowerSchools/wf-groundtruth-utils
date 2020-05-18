import numpy as np

# Thanks goes to Adrian Rosebrock, PhD, for giving me a huge head start ->
# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/


# Malisiewicz et al.
def non_max_suppression_fast(boxes, iou_thresh=0.5, max_annotations_per_object=2, prefer_highest_iou=True):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # What's the point in running max suppression if we don't have more than a single bounding box to compare
    if max_annotations_per_object < 2:
        return boxes

    # initialize the list of picked indexes
    pick = []

    idxs = np.argsort(boxes[:, 3])
    # keep looping while some indexes still remain in the indexes
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        ii = idxs[last]

        iou = intersection_over_union(boxes[idxs], last)

        # find all indexes where iou score is greater than threshold
        filtered_idxs = np.where(iou > iou_thresh)[0]

        # now sort filtered scores (argsort will arrange in ascending order so we'll pick from the end)
        filtered_idxs_sorted = filtered_idxs[np.argsort(iou[filtered_idxs])]

        # use max_annotations_per_object to set a cap on how many boxes will be removed from consideration
        num_annotations_to_remove = min(len(filtered_idxs_sorted), max_annotations_per_object - 1)
        deletable_idx = len(filtered_idxs_sorted) - num_annotations_to_remove
        filtered_idxs_trimmed = np.concatenate(([last], filtered_idxs_sorted[deletable_idx:]))

        # If there are more than 2 competing annotations, choose the annotation
        # with the highest iou when compared to all other annotations
        if prefer_highest_iou and max_annotations_per_object > 2 and len(
                filtered_idxs_trimmed) >= max_annotations_per_object:
            filtered_boxes = boxes[idxs[filtered_idxs_trimmed]]
            max_idx = 0
            max_score = 0
            for jj, _ in enumerate(filtered_boxes):
                iou = intersection_over_union(filtered_boxes, jj)
                new_max = np.amax(iou, 0)
                if new_max > max_score:
                    max_score = new_max
                    max_idx = jj
            pick.append(idxs[filtered_idxs_trimmed[max_idx]])
        else:
            pick.append(ii)

        # delete all indexes from the index list that have been matched
        idxs = np.delete(idxs, filtered_idxs_trimmed)
    # return only the bounding boxes that were picked
    return boxes[pick]


def intersection_over_union(boxes, comparable_index):
    comparables = np.delete(boxes, comparable_index, 0)

    xx1 = np.maximum(boxes[comparable_index, 0], comparables[:, 0])
    yy1 = np.maximum(boxes[comparable_index, 1], comparables[:, 1])
    xx2 = np.minimum(boxes[comparable_index, 2], comparables[:, 2])
    yy2 = np.minimum(boxes[comparable_index, 3], comparables[:, 3])

    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    iou = (w * h) / (box_areas[comparable_index] + np.delete(box_areas, comparable_index, 0) - (w * h))
    return iou
