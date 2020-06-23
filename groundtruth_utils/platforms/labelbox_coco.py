from enum import Enum, IntEnum
import re

from ..coco.models.annotation import KeypointAnnotation
from ..coco.models.category import KeypointCategory
from .labelbox_utils import labelbox_geom_to_geojson


class LabelboxOntologyType(IntEnum):
    LABELBOX_ONTOLOGY_TYPE_UNKNOWN = 0
    LABELBOX_ONTOLOGY_TYPE_BOUNDING_BOX = 1
    LABELBOX_ONTOLOGY_TYPE_KEYPOINT_VISIBILITY_FLATTENED = 2
    LABELBOX_ONTOLOGY_TYPE_KEYPOINT_VISIBILITY_NESTED = 3


class LabelboxVisibiltyType(Enum):
    LABELBOX_VISIBILITY_TYPE_UNKNOWN = ''
    LABELBOX_VISIBILITY_VISIBLE = 'Visible'
    LABELBOX_VISIBILITY_NOT_VISIBLE = 'Not Visible'


def coco_visibility_to_labelbox(coco_visibility: KeypointAnnotation.Visibility):
    if coco_visibility == KeypointAnnotation.Visibility.VISIBILITY_NOT_LABELED:
        return LabelboxVisibiltyType.LABELBOX_VISIBILITY_TYPE_UNKNOWN
    elif coco_visibility == KeypointAnnotation.Visibility.VISIBILITY_LABELED_NOT_VISIBLE:
        return LabelboxVisibiltyType.LABELBOX_VISIBILITY_NOT_VISIBLE
    elif coco_visibility == KeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE:
        return LabelboxVisibiltyType.LABELBOX_VISIBILITY_VISIBLE


def coco_category_to_labelbox(coco_category: KeypointCategory.Keypoint):
    if coco_category == KeypointCategory.Keypoint.NOSE:
        return 'Nose'
    elif coco_category == KeypointCategory.Keypoint.NECK:
        return 'Neck'
    elif coco_category == KeypointCategory.Keypoint.LEFT_EYE:
        return 'Left Eye'
    elif coco_category == KeypointCategory.Keypoint.RIGHT_EYE:
        return 'Right Eye'
    elif coco_category == KeypointCategory.Keypoint.LEFT_EAR:
        return 'Left Ear'
    elif coco_category == KeypointCategory.Keypoint.RIGHT_EAR:
        return 'Right Ear'
    elif coco_category == KeypointCategory.Keypoint.LEFT_SHOULDER:
        return 'Left Shoulder'
    elif coco_category == KeypointCategory.Keypoint.RIGHT_SHOULDER:
        return 'Right Shoulder'
    elif coco_category == KeypointCategory.Keypoint.LEFT_ELBOW:
        return 'Left Elbow'
    elif coco_category == KeypointCategory.Keypoint.RIGHT_ELBOW:
        return 'Right Elbow'
    elif coco_category == KeypointCategory.Keypoint.LEFT_WRIST:
        return 'Left Wrist'
    elif coco_category == KeypointCategory.Keypoint.RIGHT_WRIST:
        return 'Right Wrist'
    elif coco_category == KeypointCategory.Keypoint.LEFT_HIP:
        return 'Left Hip'
    elif coco_category == KeypointCategory.Keypoint.RIGHT_HIP:
        return 'Right Hip'
    elif coco_category == KeypointCategory.Keypoint.LEFT_KNEE:
        return 'Left Knee'
    elif coco_category == KeypointCategory.Keypoint.RIGHT_KNEE:
        return 'Right Knee'
    elif coco_category == KeypointCategory.Keypoint.LEFT_ANKLE:
        return 'Left Ankle'
    elif coco_category == KeypointCategory.Keypoint.RIGHT_ANKLE:
        return 'Right Ankle'


def coco_keypoint_to_labelbox_string(coco_category: KeypointCategory.Keypoint,
                                     coco_visiblity: KeypointAnnotation.Visibility):
    return "%s - %s" % (coco_category_to_labelbox(coco_category), coco_visibility_to_labelbox(coco_visiblity))


class CocoOntologyTool(object):
    def __init__(self, tool):
        self.tool = tool
        self.type = LabelboxOntologyType.LABELBOX_ONTOLOGY_TYPE_UNKNOWN
        self.visibility = None
        self.keypoint_name = None
        self.nested_classification = None

        self.__determine_tool_type()

    def __determine_tool_type(self):
        if self.tool['tool'] == 'point':
            self.__determine_keypoint_type()
        elif self.tool['tool'] == 'rectangle':
            self.type = LabelboxOntologyType.LABELBOX_ONTOLOGY_TYPE_BOUNDING_BOX

    def __determine_keypoint_type(self):
        # First test for ontology type ONTOLOGY_TYPE_KEYPOINT_VISIBILITY_FLATTENED
        r = re.compile(r'^([A-z0-9\s]+) - (Visible|Not\sVisible).*', re.IGNORECASE)
        m = r.match(self.tool['name'])
        if m is not None:
            self.type = LabelboxOntologyType.LABELBOX_ONTOLOGY_TYPE_KEYPOINT_VISIBILITY_FLATTENED

            self.keypoint_name = m[1]

            if m[2].lower() == 'visible':
                self.visibility = LabelboxVisibiltyType.LABELBOX_VISIBILITY_VISIBLE
            else:
                self.visibility = LabelboxVisibiltyType.LABELBOX_VISIBILITY_NOT_VISIBLE
            return

        # If first test fails, test for ontology type LABELBOX_ONTOLOGY_TYPE_VISIBILITY_NESTED
        for classification in self.tool['classifications']:
            if 'name' in classification and classification['name'].lower() == 'visibility':
                options = list(map(lambda o: o['label'].lower(), classification['options']))
                if sorted(options) == sorted([LabelboxVisibiltyType.LABELBOX_VISIBILITY_VISIBLE,
                                              LabelboxVisibiltyType.LABELBOX_VISIBILITY_NOT_VISIBLE]):
                    self.nested_classification = classification
                    self.type = LabelboxOntologyType.LABELBOX_ONTOLOGY_TYPE_KEYPOINT_VISIBILITY_NESTED
                    self.keypoint_name = self.tool['name']
                return

    def is_bounding_box(self):
        return self.type == LabelboxOntologyType.LABELBOX_ONTOLOGY_TYPE_BOUNDING_BOX

    def is_keypoint(self):
        return self.type == LabelboxOntologyType.LABELBOX_ONTOLOGY_TYPE_KEYPOINT_VISIBILITY_FLATTENED or \
            self.type == LabelboxOntologyType.LABELBOX_ONTOLOGY_TYPE_KEYPOINT_VISIBILITY_NESTED

    def is_coco_keypoint_conversion_to_labelbox_valid(self, coco_keypoint):
        coco_keypoint_type = KeypointCategory.Keypoint(self.keypoint_name)

        # First be sure the coco keypoint being converted to Labelbox exists, otherwise return None
        coco_visibility = coco_keypoint.get_keypoint_visibility(coco_keypoint_type)
        if coco_visibility == KeypointAnnotation.Visibility.VISIBILITY_NOT_LABELED:
            return False

        # Next, if working with the flattened ontology type, check if
        # coco_visibility aligns with this ontology tool visibility
        if self.type == LabelboxOntologyType.LABELBOX_ONTOLOGY_TYPE_KEYPOINT_VISIBILITY_FLATTENED:
            if coco_visibility_to_labelbox(coco_visibility) == LabelboxVisibiltyType.LABELBOX_VISIBILITY_VISIBLE and \
                    self.visibility != LabelboxVisibiltyType.LABELBOX_VISIBILITY_VISIBLE:
                return False
            elif coco_visibility_to_labelbox(coco_visibility) == LabelboxVisibiltyType.LABELBOX_VISIBILITY_NOT_VISIBLE and \
                    self.visibility != LabelboxVisibiltyType.LABELBOX_VISIBILITY_NOT_VISIBLE:
                return False

        return True

    def to_labelbox_label(self, coco_keypoint):
        label = {
            "schemaId": self.tool['featureSchemaId'],
            "title": self.tool['name'],
            "value": re.compile(r'\s+').sub("_", self.tool['name']).lower(),
            "color": self.tool['color'],
            "instanceURI": "https://api.labelbox.com/masks/feature/%s" % (self.tool['featureSchemaId'])
        }

        if self.is_bounding_box():
            coco_box = coco_keypoint.get_bounding_box()
            label['bbox'] = {
                "left": coco_box[0],
                "top": coco_box[1],
                "width": coco_box[2],
                "height": coco_box[3]
            }

        elif self.is_keypoint():
            if not self.is_coco_keypoint_conversion_to_labelbox_valid(coco_keypoint):
                return None

            coco_keypoint_type = KeypointCategory.Keypoint(self.keypoint_name)
            coco_point = coco_keypoint.get_keypoint_point(coco_keypoint_type)
            label['point'] = {
                "x": coco_point[0],
                "y": coco_point[1]
            }

            if self.type == LabelboxOntologyType.LABELBOX_ONTOLOGY_TYPE_KEYPOINT_VISIBILITY_NESTED:
                nested_classification_answer = {}

                nested_option = None
                if coco_keypoint.is_keypoint_visible():
                    nested_option = list(
                        filter(
                            lambda o: o['label'].lower() == LabelboxVisibiltyType.LABELBOX_VISIBILITY_NOT_VISIBLE.value),
                        self.classification['options'])
                elif coco_keypoint.is_keypoint_not_visible():
                    nested_option = list(
                        filter(
                            lambda o: o['label'].lower() == LabelboxVisibiltyType.LABELBOX_VISIBILITY_VISIBLE.value),
                        self.classification['options'])

                if nested_option:
                    nested_classification_answer = {
                        "schemaId": nested_option['featureSchemaId'],
                        "title": nested_option['label'],
                        "value": nested_option['value']
                    }

                label['classifications'] = [
                    {
                        "schemaId": self.nested_classification['featureSchemaId'],
                        "title": self.nested_classification['instructions'],
                        "value": self.nested_classification['name'],
                        "answer": nested_classification_answer
                    }
                ]

        return label

    def get_nested_classification(self, coco_keypoint):
        if not self.is_keypoint():
            return None

        if not self.is_coco_keypoint_conversion_to_labelbox_valid(coco_keypoint):
            return None

        if self.type != LabelboxOntologyType.LABELBOX_ONTOLOGY_TYPE_KEYPOINT_VISIBILITY_NESTED:
            return None

        nested_option = None
        if coco_keypoint.is_keypoint_visible():
            nested_option = list(
                filter(
                    lambda o: o['label'].lower() == LabelboxVisibiltyType.LABELBOX_VISIBILITY_NOT_VISIBLE.value),
                self.classification['options'])
        elif coco_keypoint.is_keypoint_not_visible():
            nested_option = list(
                filter(
                    lambda o: o['label'].lower() == LabelboxVisibiltyType.LABELBOX_VISIBILITY_VISIBLE.value),
                self.classification['options'])

        if nested_option is None:
            return None

        return {
            'question_schema_id': self.nested_classification['featureSchemaId'],
            'options_schema_ids': [nested_option['featureSchemaId']]
        }

    def to_labelbox_geometry_format(self, coco_keypoint):
        if self.is_bounding_box():
            coco_box = coco_keypoint.get_bounding_box()
            return {
                "bbox": {
                    "left": coco_box[0],
                    "top": coco_box[1],
                    "width": coco_box[2],
                    "height": coco_box[3]
                }
            }
        elif self.is_keypoint():
            if not self.is_coco_keypoint_conversion_to_labelbox_valid(coco_keypoint):
                return None

            coco_keypoint_type = KeypointCategory.Keypoint(self.keypoint_name)
            coco_point = coco_keypoint.get_keypoint_point(coco_keypoint_type)

            return {
                "point": {
                    "x": coco_point[0],
                    "y": coco_point[1]
                }
            }
        else:
            return None


def coco_annotation_to_labelbox(coco_annotation: KeypointAnnotation, normalized_ontology):
    labels = []

    for tool in normalized_ontology['tools']:
        coco_ontology_tool = CocoOntologyTool(tool)

        if coco_ontology_tool.type != LabelboxOntologyType.LABELBOX_ONTOLOGY_TYPE_UNKNOWN:
            labelbox_geom = coco_ontology_tool.to_labelbox_geometry_format(coco_annotation)
            if labelbox_geom is None:
                continue

            labels.append({
                'schema_id': tool['featureSchemaId'],
                'labelbox_geom': labelbox_geom,
                'geo_json': labelbox_geom_to_geojson(labelbox_geom),
                'nested_classification_feature': coco_ontology_tool.get_nested_classification(coco_annotation)
            })

    return labels
