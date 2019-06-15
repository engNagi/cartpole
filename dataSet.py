import numpy as np
from skimage.measure import block_reduce
from skimage.transform import resize


class dataSet(object):
    def __init__(self, frame=None, state_height=None, state_width=None):
        self.frame = frame
        self.stacked_state = []
        self.image_height = state_height
        self.image_width = state_width

    def state_to_greyscale(self, frame):
        return np.mean(frame, axis=2).astype(np.uint8)

    def state_resize(self, frame, frame_height, frame_width):
        return resize(frame, (frame_height, frame_width), preserve_range=True).astype(np.uint8)

