import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

tf.get_logger().setLevel('ERROR')
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from object_detection.utils import config_util
import glob
from object_detection.builders import model_builder
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import matplotlib
import warnings
import random
from PIL import Image

warnings.filterwarnings('ignore')
matplotlib.use('tkagg')


class Model:
    """
    Wraps tensorflow object detections models
    Mainly restores models and detects objects
    """

    def __init__(self, ckpt_path, ckpt_index, label_map_path, max_detections):
        self.ckpt_path = ckpt_path
        self.label_map_path = label_map_path
        self.pipeline_path = self._find_config()
        self.max_detections = max_detections
        self.as_tf_model = self._restore_model(ckpt_index)
        self.category_index = label_map_util.create_category_index_from_labelmap(
            self.label_map_path,
            use_display_name=True
        )
        self.datasets = []
        self.min_score = .8
        self.num_detections = 0
        self._warmup()

    @property
    def ckpt_path(self):
        return self._ckpt_path

    @ckpt_path.setter
    def ckpt_path(self, ckpt_path):
        self._ckpt_path = self._valid_dir(ckpt_path)

    @property
    def label_map_path(self):
        return self._label_map_path

    @label_map_path.setter
    def label_map_path(self, label_map_path):
        self._label_map_path = self._valid_file(label_map_path)

    @property
    def max_detections(self):
        return self._max_detections

    @max_detections.setter
    def max_detections(self, max_detections):
        if max_detections < 1:
            raise ValueError("Value must be positive")
        if max_detections > 5:
            print("Warning... It is recommended to use 5 as a max value with 16GB RAM")
        self._max_detections = max_detections

    def _warmup(self):
        """
        Model warmup as ...
        'The TensorFlow runtime has components that are lazily initialized'
        """
        for img in random.sample(
                glob.glob(
                    "..\\test\\Limit*.jpg"),
                5):
            self.get_detections(np.array(Image.open(img)))

    @staticmethod
    def _valid_file(path):
        """
        A setter for file paths
        """
        if os.path.isfile(path):
            return path
        raise FileNotFoundError("{} is not a file".format(path))

    @staticmethod
    def _valid_dir(path):
        """
        A setter for dir paths
        """
        if os.path.isdir(path):
            return path
        raise FileNotFoundError("{} is not a dir".format(path))

    def _find_config(self):
        files = glob.glob(os.path.join(self.ckpt_path, "*.config"))
        if len(files) > 0:
            return files[0]
        raise FileNotFoundError("Make sure there's a pipeline configuration inside {}".format(self.ckpt_path))

    def _restore_model(self, index):
        """
        Restore the i[th] version of the model from its checkpoint path
        params:
            [1] index: checkpoint index (probably 3)
        return:
            detection model object
        """
        configs = config_util.get_configs_from_pipeline_file(self.pipeline_path)
        detection_model = model_builder.build(model_config=configs['model'], is_training=False)
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(self.ckpt_path, "ckpt-{}".format(index))).expect_partial()
        return detection_model

    @tf.function
    def _detect_fn(self, image):
        """
        Tensorflow function to get postprocess detections
        params:
            [1] image: image array
        return:
            detections
        """
        image, shapes = self.as_tf_model.preprocess(image)
        prediction_dict = self.as_tf_model.predict(image, shapes)
        detections = self.as_tf_model.postprocess(prediction_dict, shapes)
        return detections

    def get_tf_detections(self, image_np):
        """
        Pass the input tensor to tensorflow detect function
        params:
            [1] image: image array
        return:
            detections
        """
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = self._detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        return detections

    def get_detections(self, image_np, tf_detections=None):
        """
        Get detections as a simple readable format {label: score}
        This is not a vizualization function but it was created in visuz
        params:
            [1] image array
        return:
            detections dictionary
        """
        if not tf_detections:
            detections = self.get_tf_detections(image_np)
        else:
            detections = tf_detections
        label_id_offset = 1
        return viz_utils.get_detections(
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset,
            detections['detection_scores'],
            self.category_index,
            min_score_thresh=self.min_score
        )

    def get_image_np_with_detections(self, image_np, tf_detections=None):
        """

        """
        if not tf_detections:
            detections = self.get_tf_detections(image_np)
        else:
            detections = tf_detections
        label_id_offset = 1
        image_np_with_detections = image_np.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset,
            detections['detection_scores'],
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=self.min_score,
            agnostic_mode=False)
        return image_np_with_detections

    def is_allowed(self):
        return self.num_detections <= self.max_detections
