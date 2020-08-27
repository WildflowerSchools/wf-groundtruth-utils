import re
import requests
import tempfile
import time
import yaml

from jsonpath_ng.ext import parse

from .annotate import Annotate
from .coco.models.annotation import KeypointAnnotation as CocoKeypointAnnotation
from .coco.models.coco import Coco
from .coco.models.category import KeypointCategory as CocoKeypointCategory, all_coco_categories, get_coco_category
from .coco.models.image import Image as CocoImage
from .helper import *
from .log import logger


class CocoGenerator:
    def __init__(self):
        self.coco = Coco(
            categories=all_coco_categories()
        )

    @staticmethod
    def load_config(config_file):
        return yaml.load(config_file, Loader=yaml.FullLoader)

    def load_data_from_platform(self, platform, config_file, separate_by_annotation=False,
                                filter_min_confidence=0.0, filter_min_labelers=3):
        config = self.__class__.load_config(config_file)

        coco_images = {}
        for job_config in config['jobs']:
            logger.info("Loading '%s' annotations" % job_config['name'])
            active_platform = get_platform(platform)
            valid_images, invalid_images = active_platform.fetch_annotations(
                job_config['name'],
                consolidate=True,
                filter_min_confidence=filter_min_confidence,
                filter_min_labelers=filter_min_labelers)

            valid_images.set_excluded_null()

            incomplete_image_ids = []
            for image_idx, image in enumerate(invalid_images.images):
                external_id = image.external_id
                if 'externalIdPattern' in job_config:
                    r = re.compile(job_config['externalIdPattern'])
                    external_id = ''.join(re.split(r, external_id))
                incomplete_image_ids.append(external_id)

            image_id = 0
            for image_idx, image in enumerate(valid_images.images):
                logger.info("%s - Generating annotations" % image.external_id)

                image_as_dict = image.dict()
                external_id = image.external_id
                if 'externalIdPattern' in job_config:
                    r = re.compile(job_config['externalIdPattern'])
                    external_id = ''.join(re.split(r, external_id))

                if external_id in incomplete_image_ids:
                    logger.info(
                        "%s - Skipping because image still has pending annotations that have not been completed or match filter rules" %
                        (image.external_id))
                    continue

                annotation_idx = 0
                all_annotations = []
                bbox_category = None
                for annotation_config in job_config['annotations']:
                    if annotation_config['type'] == 'bbox':
                        logger.info(
                            "%s - Parsing bbox annotations for category '%s'" %
                            (image.external_id, annotation_config['category']))
                        jsonpath_expr = parse(annotation_config['match'])

                        for idx, match in enumerate(jsonpath_expr.find(image_as_dict['annotations'])):
                            if not separate_by_annotation and bbox_category is not None:
                                logger.warning(
                                    "%s - Combine mode expects a single bbox, multiple bboxes ignored - %s" %
                                    (image.external_id, annotation_config['category']))
                                continue

                            bbox_category = annotation_config['category']
                            all_annotations += [{
                                'annotation': match.value,
                                'type': 'bbox',
                                'category': annotation_config['category'],
                                'visibility': True
                            }]
                    elif annotation_config['type'] == 'keypoint':
                        if separate_by_annotation:
                            logger.warning(
                                "%s - 'Separate' by annotation mode ignores keypoints, passing on %s" %
                                (image.external_id, annotation_config['category']))
                            continue

                        logger.info("%s - Parsing keypoint visible annotations for category '%s'" %
                                    (image.external_id, annotation_config['category']))
                        jsonpath_expr = parse(annotation_config['visible'])
                        visible_annotations = [
                            {
                                'annotation': match.value,
                                'type': 'keypoint',
                                'category': annotation_config['category'],
                                'visibility': CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE} for match in jsonpath_expr.find(
                                image_as_dict['annotations'])]

                        logger.info("%s - Parsing keypoint not-visible annotations for category '%s'" %
                                    (image.external_id, annotation_config['category']))
                        jsonpath_expr = parse(annotation_config['notVisible'])
                        not_visible_annotations = [
                            {
                                'annotation': match.value,
                                'type': 'keypoint',
                                'category': annotation_config['category'],
                                'visibility': CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_NOT_VISIBLE} for match in jsonpath_expr.find(
                                image_as_dict['annotations'])]

                        all_annotations += visible_annotations + not_visible_annotations

                for annotation_match_idx, annotation_match in enumerate(all_annotations):
                    file_name = external_id
                    if separate_by_annotation:
                        file_name = get_separated_file_name(file_name, annotation_idx)

                    if file_name not in coco_images:
                        image_id += 1
                        coco_images[file_name] = {
                            'image': CocoImage(
                                id=image_id,
                                file_name=file_name,
                                coco_url=os.path.join(os.path.split(image.url)[0], file_name),
                                width=0,
                                height=0),
                            'annotations': {}}

                    if separate_by_annotation:
                        annotation_idx += 1

                    external_annotation_id = image.external_id
                    if 'annotationIdPattern' in job_config:
                        rexp = re.compile(job_config['annotationIdPattern'])
                        external_annotation_id = "%s - %s" % (file_name,
                                                              ''.join(re.split(rexp, image.external_id)))
                    elif separate_by_annotation:
                        external_annotation_id = "%s - %s" % (file_name, annotation_match_idx)

                    if external_annotation_id not in coco_images[file_name]['annotations']:
                        if separate_by_annotation:
                            category_id = get_coco_category(annotation_match['category']).id
                        else:
                            category_id = get_coco_category(bbox_category).id

                        coco_images[file_name]['annotations'][external_annotation_id] = CocoKeypointAnnotation(
                            image_id=image_id, category_id=category_id)

                    current_annotation = coco_images[file_name]['annotations'][external_annotation_id]
                    if annotation_match['type'] == 'bbox':
                        current_annotation.bbox = [
                            annotation_match['annotation']['left'],
                            annotation_match['annotation']['top'],
                            annotation_match['annotation']['width'],
                            annotation_match['annotation']['height']]
                    elif annotation_match['type'] == 'keypoint':
                        current_annotation.add_keypoint(
                            CocoKeypointCategory.Keypoint(annotation_match['category']),
                            annotation_match['annotation']['x'],
                            annotation_match['annotation']['y'],
                            CocoKeypointAnnotation.Visibility(annotation_match['visibility']))

        annotation_id = 0
        for external_id in coco_images:
            self.coco.images.append(coco_images[external_id]['image'])
            for external_annotation_id in coco_images[external_id]['annotations']:
                coco_images[external_id]['annotations'][external_annotation_id].id = annotation_id
                coco_images[external_id]['annotations'][external_annotation_id].compute_area()
                coco_images[external_id]['annotations'][external_annotation_id].compute_num_keypoints()
                self.coco.annotations.append(coco_images[external_id]['annotations'][external_annotation_id])
                annotation_id += 1

    def load_data_with_classifiers(self, image_urls):
        annotator = Annotate()

        for image_idx, image in enumerate(image_urls):
            logger.info("Downloading image %s" % (image.url))
            tic = time.time()
            response = requests.get(image.url)
            if not response.ok:
                logger.warn("Failed downloading %s" % (image.url))
                continue

            logger.info('Done Downloading (t={:0.2f}s)'.format(time.time() - tic))

            with tempfile.NamedTemporaryFile() as temp_image:
                temp_image.write(response.content)
                temp_image.flush()

                logger.info("Annotating image %s" % (image.url))
                tic = time.time()
                annotations = annotator.annotate_image(temp_image.name)
                if annotations is None:
                    logger.error('Annotation returned unexpected result, exiting')
                    return None

                coco_annotations = []
                for annotation in annotations:
                    coco_keypoint = CocoKeypointAnnotation(
                        image_id=image_idx,
                        bbox=annotation['bbox'],
                        category_id=CocoKeypointCategory.coco_person_category().id)
                    coco_keypoint.add_keypoint(
                        CocoKeypointCategory.Keypoint.NOSE,
                        annotation['keypoints'][0][0],
                        annotation['keypoints'][0][1],
                        CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                    coco_keypoint.add_keypoint(
                        CocoKeypointCategory.Keypoint.LEFT_EYE,
                        annotation['keypoints'][1][0],
                        annotation['keypoints'][1][1],
                        CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                    coco_keypoint.add_keypoint(
                        CocoKeypointCategory.Keypoint.RIGHT_EYE,
                        annotation['keypoints'][2][0],
                        annotation['keypoints'][2][1],
                        CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                    coco_keypoint.add_keypoint(
                        CocoKeypointCategory.Keypoint.LEFT_EAR,
                        annotation['keypoints'][3][0],
                        annotation['keypoints'][3][1],
                        CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                    coco_keypoint.add_keypoint(
                        CocoKeypointCategory.Keypoint.RIGHT_EAR,
                        annotation['keypoints'][4][0],
                        annotation['keypoints'][4][1],
                        CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                    coco_keypoint.add_keypoint(
                        CocoKeypointCategory.Keypoint.LEFT_SHOULDER,
                        annotation['keypoints'][5][0],
                        annotation['keypoints'][5][1],
                        CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                    coco_keypoint.add_keypoint(
                        CocoKeypointCategory.Keypoint.RIGHT_SHOULDER,
                        annotation['keypoints'][6][0],
                        annotation['keypoints'][6][1],
                        CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                    coco_keypoint.add_keypoint(
                        CocoKeypointCategory.Keypoint.LEFT_ELBOW,
                        annotation['keypoints'][7][0],
                        annotation['keypoints'][7][1],
                        CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                    coco_keypoint.add_keypoint(
                        CocoKeypointCategory.Keypoint.RIGHT_ELBOW,
                        annotation['keypoints'][8][0],
                        annotation['keypoints'][8][1],
                        CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                    coco_keypoint.add_keypoint(
                        CocoKeypointCategory.Keypoint.LEFT_WRIST,
                        annotation['keypoints'][9][0],
                        annotation['keypoints'][9][1],
                        CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                    coco_keypoint.add_keypoint(
                        CocoKeypointCategory.Keypoint.RIGHT_WRIST,
                        annotation['keypoints'][10][0],
                        annotation['keypoints'][10][1],
                        CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                    coco_keypoint.add_keypoint(
                        CocoKeypointCategory.Keypoint.LEFT_HIP,
                        annotation['keypoints'][11][0],
                        annotation['keypoints'][11][1],
                        CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                    coco_keypoint.add_keypoint(
                        CocoKeypointCategory.Keypoint.RIGHT_HIP,
                        annotation['keypoints'][12][0],
                        annotation['keypoints'][12][1],
                        CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                    coco_keypoint.add_keypoint(
                        CocoKeypointCategory.Keypoint.LEFT_KNEE,
                        annotation['keypoints'][13][0],
                        annotation['keypoints'][13][1],
                        CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                    coco_keypoint.add_keypoint(
                        CocoKeypointCategory.Keypoint.RIGHT_KNEE,
                        annotation['keypoints'][14][0],
                        annotation['keypoints'][14][1],
                        CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                    coco_keypoint.add_keypoint(
                        CocoKeypointCategory.Keypoint.LEFT_ANKLE,
                        annotation['keypoints'][15][0],
                        annotation['keypoints'][15][1],
                        CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                    coco_keypoint.add_keypoint(
                        CocoKeypointCategory.Keypoint.RIGHT_ANKLE,
                        annotation['keypoints'][16][0],
                        annotation['keypoints'][16][1],
                        CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                    coco_keypoint.add_keypoint(
                        CocoKeypointCategory.Keypoint.NECK,
                        annotation['keypoints'][17][0],
                        annotation['keypoints'][17][1],
                        CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)

                    coco_annotations.append(coco_keypoint)

                self.coco.images.append(
                    CocoImage(
                        id=image_idx,
                        file_name=os.path.basename(
                            image.url),
                        coco_url=image.url,
                        width=0,
                        height=0))
                self.coco.annotations.extend(coco_annotations)

                logger.info('Done Annotating (t={:0.2f}s)'.format(time.time() - tic))

    def load_data_from_pycoco(self, pycoco):
        self.coco.load_pycoco(pycoco)

    def model(self):
        return self.coco
