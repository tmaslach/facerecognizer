import cv2
import numpy as np
import os

def to_fullpath(filename):
    this_dir = os.path.dirname(__file__)
    return os.path.join(this_dir, filename)    

class FaceDetector:
    caffee_prototxt = to_fullpath("data/deploy.prototxt")
    caffee_model    = to_fullpath("data/res10_300x300_ssd_iter_140000_fp16.caffemodel")

    class Detection:
        def __init__ (self, confidence, start_x, start_y, end_x, end_y):
            self.confidence = confidence
            self.start_x = start_x
            self.start_y = start_y
            self.end_x = end_x
            self.end_y = end_y

    def __init__ (self, confidence=.5):
        self.confidence_needed = confidence
        self.net = cv2.dnn.readNetFromCaffe(self.caffee_prototxt, self.caffee_model)

    def scan(self, image):
        height, width, _ = image.shape

        resized_image = cv2.resize(image, (300, 300))
        blob = cv2.dnn.blobFromImage(resized_image,
            scalefactor = 1.0,
            size        = (300, 300),
            mean        = (104.0, 177.0, 123.0) 
        )

        self.net.setInput(blob)
        output = self.net.forward()
        
        detections = []
        for i in range(output.shape[2]):
            confidence = output[0, 0, i, 2]
            if confidence >= self.confidence_needed:
                box = output[0, 0, i, 3:7] * np.array([width, height, width, height])
                start_x, start_y, end_x, end_y = box.astype("int")
                detections.append(self.Detection(confidence, start_x, start_y, end_x, end_y))

        return detections
