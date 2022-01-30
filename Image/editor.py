import cv2
import numpy as np
from collections import namedtuple

CircleCrop = namedtuple("circle", ["minDist", "dp", "param1", "param2", "minRadius", "maxRadius"])


class FilterKeys:
    """
    Acceptable filter keys
    that won't raise key error
    """
    CROP = "crop"
    SHAPE = "shape"
    COLOR = "color"
    SIZE = "size"
    CORDS = "cords"


class Filter:
    """
    Clarifications
    --------------
    CROP clockwise [left, top, right, bottom]
    CORDS are top left pixels of desired search areas (by size)
    """
    KEYS = [FilterKeys.CROP, FilterKeys.SHAPE,
            FilterKeys.COLOR, FilterKeys.SIZE,
            FilterKeys.CORDS]

    def __init__(self, dct=()):
        self._as_dict = {}
        for key in dct:
            self[key] = dct[key]

    def __getitem__(self, item):
        return self._as_dict[item]

    def __setitem__(self, key, value):
        if key not in self.KEYS:
            raise KeyError(f"Acceptable keys are {'|'.join(self.KEYS)}")
        self._as_dict[key] = value

    def get_dict(self):
        return self._as_dict


class Filters:
    """
    Speed-limit crop
    * generates smaller inputs for search-areas crops
    """
    SPEED_LIMIT_CROP = Filter(
        {
            FilterKeys.CROP: (0, 600, 300, 800)
        }
    )
    """
    Speed-limit search area
    * generates smaller inputs for detection
    """
    SPEED_LIMIT_SEARCH_AREA = Filter(
        {
            FilterKeys.SIZE: (300, 350),
            FilterKeys.CORDS: ((0, 0), (150, 0), (300, 0),
                               (0, 150), (150, 150), (300, 150),
                               (150, 150), (200, 100), (400, 200))
        }
    )


class ImageEditor:
    def __init__(self, path=None, img=None):
        self.img = img
        if path:
            self.img = cv2.imread(path)
        self.circle_crop = CircleCrop(minDist=0, dp=0, param1=0,
                                      param2=0, minRadius=0, maxRadius=1)

    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, img):
        self._img = img

    @staticmethod
    def _adjust_image_box(box, shape):
        left, top, right, bottom = box
        while left < 0:
            left, right = left + 1, right + 1
        while right > shape[1]:
            left, right = left - 1, right - 1
        while top < 0:
            top, bottom = top + 1, bottom + 1
        while bottom > shape[0]:
            top, bottom = top - 1, bottom - 1
        return left, top, right, bottom

    def crop_circle_box(self, image=None):
        image = self.img if image is None else image
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray_img,
                                   cv2.HOUGH_GRADIENT,
                                   minDist=self.circle_crop.minDist,
                                   dp=self.circle_crop.dp,
                                   param1=self.circle_crop.param1,
                                   param2=self.circle_crop.param2,
                                   minRadius=self.circle_crop.minRadius,
                                   maxRadius=self.circle_crop.maxRadius)
        ret = []
        if circles is not None:
            for i in range(len(circles[0])):
                circles = np.uint16(np.around(circles))
                x, y, r = (int(x) for x in circles[0][i])
                left, top, right, bottom = self._adjust_image_box(
                    box=(x - 100, y - 100, x + 100, y + 100),
                    shape=image.shape
                )
                ret.append((image[top:bottom, left:right], (x, y, r)))
        return ret

    def crop_search_areas(self, size, cords, function):
        for cord in cords:
            for processed_image in function(
                    self.img[cord[0]:cord[0] + size[0], cord[1]:cord[1] + size[1]]):
                yield processed_image

    def implement_filter(self, im_filter):
        if FilterKeys.COLOR in im_filter.get_dict():
            self.img = self.img.convert(im_filter[FilterKeys.COLOR])

        if FilterKeys.CROP in im_filter.get_dict():
            top, bottom, left, right = im_filter[FilterKeys.CROP]
            self.img = self.img[top:bottom, left:right]

        if FilterKeys.SHAPE in im_filter.get_dict():
            self.img = cv2.resize(self.img, dsize=FilterKeys.SHAPE)

        return self.img


class SpeedLimitEditor(ImageEditor):
    def __init__(self):
        super().__init__()
        self.circle_crop = CircleCrop(minDist=6, dp=1.1, param1=150,
                                      param2=50, minRadius=20, maxRadius=50)
