import cv2
import random
from utils import timeit

cfg = "models/coco_model/yolov4-tiny.cfg"
weights = "models/coco_model/yolov4-tiny.weights"
className = "models/coco.names"

class Detector:
    """Class ssd"""

    @timeit
    def __init__(self):
        self.net = cv2.dnn_DetectionModel(cfg, weights)
        self.net.setInputSize(416, 416)
        self.net.setInputScale(1.0 / 255)
        self.net.setInputSwapRB(True)
        with open(className, 'rt') as f:
            self.names = f.read().rstrip('\n').split('\n')

        self.thread = None
        self.img = None
        self.is_processing = False

    @timeit
    def detect(self, img):
        if self.is_processing:
            return

        self.is_processing = True

        classes, confidences, boxes = self.net.detect(img, confThreshold=0.1, nmsThreshold=0.4)
        if len(confidences) > 0:
            for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                label = '%s: %.2f' % (self.names[classId], confidence)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                left, top, width, height = box
                top = max(top, labelSize[1])
                # b = random.randint(0, 255)
                # g = random.randint(0, 255)
                # r = random.randint(0, 255)
                b = 0
                g = 0
                r = 255
                cv2.rectangle(img, box, color=(b, g, r), thickness=2)
                cv2.rectangle(img, (left - 1, top - labelSize[1]), (left + labelSize[0], top), (b, g, r), cv2.FILLED)
                cv2.putText(img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255 - b, 255 - g, 255 - r))

        self.img = img

        self.is_processing = False
        # self.thread.stop()