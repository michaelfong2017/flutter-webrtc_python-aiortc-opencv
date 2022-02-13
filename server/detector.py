import json
import cv2 as cv
import numpy as np
import pandas as pd
from utils import timeit, draw_boxed_text

# Initialize the parameters
confThreshold = 0.1  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image  yolov4: 608, yolov4-tiny: 416
inpHeight = 416  # Height of network's input image

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "models/coco_model/yolov4-tiny.cfg"
modelWeights = "models/coco_model/yolov4-tiny.weights"

# Load names of classes
classesFile = "models//coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(classes))]

class Detector:
    """Class ssd"""

    @timeit
    def __init__(self):
        self.net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    # Get the names of the output layers
    @timeit
    def getOutputsNames(self, net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # print(dir(net))
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Draw the predicted bounding box
    @timeit
    def drawPred(self, img, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv.rectangle(img, (left, top), (right, bottom), (0,0,255), thickness=4)

        label = '%.2f' % conf

        # Get the label for the class name and its confidence
        if classes:
            assert (classId < len(classes))
            label = '%s:%s' % (classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv.putText(img, label, (left, top-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)

    # Remove the bounding boxes with low confidence using non-maxima suppression
    @timeit
    def postprocess(self, img, outs):
        frameHeight = img.shape[0]
        frameWidth = img.shape[1]

        classIds = []
        confidences = []
        boxes = []
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.drawPred(img, classIds[i], confidences[i], left, top, left + width, top + height)
