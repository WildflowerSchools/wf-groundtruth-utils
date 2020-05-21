import re
import yaml

from jsonpath_ng.ext import parse
from .log import logger


from .coco.models.annotation import KeypointAnnotation
from .coco.models.coco import Coco
from .coco.models.category import KeypointCategory
from .coco.models.image import Image
from .helper import *


class CocoGenerator:
    def __init__(self, config, platform):
        self.raw_config = config
        self.config = self.__class__.load_config(config)
        self.platform = platform
        self.person_keypoint_category = KeypointCategory.coco_person_category()
        self.coco = Coco(
            categories=[self.person_keypoint_category]
        )
        self.coco_images = {}

    @staticmethod
    def load_config(config_file):
        return yaml.load(config_file, Loader=yaml.FullLoader)

    def load_data(self):
        for job_config in self.config['jobs']:
            logger.info("Loading '%s' annotations" % job_config['name'])
            active_platform = get_platform(self.platform)
            images = active_platform.fetch_annotations(job_config['name'], True)
            images.set_excluded_null()

            for image_idx, image in enumerate(images.images):
                external_id = image.external_id
                image_as_dict = image.dict()
                if 'externalIdPattern' in job_config:
                    r = re.compile(job_config['externalIdPattern'])
                    external_id = ''.join(re.split(r, external_id))

                if external_id not in self.coco_images:
                    self.coco_images[external_id] = {
                        'image': Image(id=image_idx, file_name=external_id, width=0, height=0),
                        'annotations': {}}

                for annotation_config in job_config['annotations']:
                    all_annotations = []

                    if annotation_config['type'] == 'bbox':
                        logger.info(
                            "%s:%s - Fetching bbox annotations for category '%s'" %
                            (image.external_id, external_id, annotation_config['category']))
                        jsonpath_expr = parse(annotation_config['match'])
                        all_annotations += [{'annotation': match.value, 'idx': idx}
                                            for idx, match in enumerate(jsonpath_expr.find(image_as_dict['annotations']))]
                        pass

                    elif annotation_config['type'] == 'keypoint':
                        logger.info("%s:%s - Fetching keypoint visible annotations for category '%s'" %
                                    (image.external_id, external_id, annotation_config['category']))
                        jsonpath_expr = parse(annotation_config['visible'])
                        visible_annotations = [
                            {
                                'annotation': match.value,
                                'visibility': KeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE} for match in jsonpath_expr.find(
                                image_as_dict['annotations'])]

                        logger.info("%s:%s - Fetching keypoint not-visible annotations for category '%s'" %
                                    (image.external_id, external_id, annotation_config['category']))
                        jsonpath_expr = parse(annotation_config['notVisible'])
                        not_visible_annotations = [
                            {
                                'annotation': match.value,
                                'visibility': KeypointAnnotation.Visibility.VISIBILITY_LABELED_NOT_VISIBLE} for match in jsonpath_expr.find(
                                image_as_dict['annotations'])]

                        logger.info("Merging annotations into data structure")
                        all_annotations += visible_annotations + not_visible_annotations

                    for annotation_match in all_annotations:
                        external_annotation_id = image.external_id
                        if 'annotationIdPattern' in job_config:
                            rexp = re.compile(job_config['annotationIdPattern'])
                            external_annotation_id = "%s - %s" % (external_id,
                                                                  ''.join(re.split(rexp, image.external_id)))
                        elif annotation_config['type'] == 'bbox':
                            external_annotation_id = "%s - %s" % (external_id, annotation_match['idx'])

                        if external_annotation_id not in self.coco_images[external_id]['annotations']:
                            self.coco_images[external_id]['annotations'][external_annotation_id] = KeypointAnnotation(
                                image_id=image_idx, category_id=self.person_keypoint_category.id)

                        if annotation_config['type'] == 'bbox':
                            self.coco_images[external_id]['annotations'][external_annotation_id].bbox = [
                                annotation_match['annotation']['left'],
                                annotation_match['annotation']['top'],
                                annotation_match['annotation']['width'],
                                annotation_match['annotation']['height']]
                        elif annotation_config['type'] == 'keypoint':
                            self.coco_images[external_id]['annotations'][external_annotation_id].add_keypoint(
                                annotation_config['category'],
                                annotation_match['annotation']['x'],
                                annotation_match['annotation']['y'],
                                annotation_match['visibility'])

    def model(self):
        logger.info("Generating COCO model")
        annotation_id = 0
        for external_id in self.coco_images:
            self.coco.images.append(self.coco_images[external_id]['image'])
            for external_annotation_id in self.coco_images[external_id]['annotations']:
                self.coco_images[external_id]['annotations'][external_annotation_id].id = annotation_id
                self.coco_images[external_id]['annotations'][external_annotation_id].compute_area()
                self.coco_images[external_id]['annotations'][external_annotation_id].compute_num_keypoints()
                self.coco.annotations.append(self.coco_images[external_id]['annotations'][external_annotation_id])
                annotation_id += 1

        return self.coco
