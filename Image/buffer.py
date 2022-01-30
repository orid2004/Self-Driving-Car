import queue
import uuid
import os

from Image.editor import *


class ImageBuffer:
    def __init__(self, path):
        self._buffer_queue = queue.Queue()
        self.path = path
        self.editor = ImageEditor()
        if not os.path.isdir(self.path):
            os.makedirs(self.path)

    def add(self, image):
        self._buffer_queue.put(image)

    def save_to_disk(self, label):
        sub_dir = 1
        while os.path.isdir(os.path.join(self.path, str(sub_dir))):
            sub_dir += 1
        os.makedirs(os.path.join(self.path, str(sub_dir)))
        while self._buffer_queue.qsize() > 0:
            self._buffer_queue.get().save(
                os.path.join(self.path, str(sub_dir), "{}-{}.jpg".format(label, uuid.uuid1())))