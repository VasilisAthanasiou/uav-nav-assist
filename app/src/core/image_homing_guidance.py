import cv2 as cv
import numpy as np
import os
import imutils
import time


class Detector:
    def __init__(self, net, labels, blob_size=(320, 320)):
        self.net = net
        self.image = None
        self.labels = labels
        self.blob_size = blob_size
        # Create a list of colors that will be used for bounding boxes
        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")
        self.start_time = time.time()

    def detect(self, image):

        """

        Args:
            net: Neural network loaded from cv2.dnn
            image: Image in which object detection will perform on
            labels: Possible labels for objects on image

        Returns:

        """
        # Initialize image
        self.image = image
        height, width = self.image.shape[:2]

        # Determine only the *output* layer names needed from YOLO
        layer_names = self.net.getLayerNames()
        layer_names = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # Construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv.dnn.blobFromImage(self.image, 1 / 255.0, self.blob_size, swapRB=True, crop=False)
        self.net.setInput(blob)

        layer_outputs = self.net.forward(layer_names)

        # Initialize lists of detected bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []

        # Loop over each of the layer outputs
        for output in layer_outputs:
            # Loop over each of the detections
            for detection in output:

                # Extract the class ID and confidence (i.e., probability) of the current object detection
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Filter out weak predictions by ensuring the detected probability is greater than the minimum probability
                if confidence > 0.5:

                    # Scale the bounding box coordinates back relative to the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of the bounding box followed by the boxes' width and height
                    box = detection[0:4] * np.array([width, height, width, height])
                    (center_x, center_y, width, height) = box.astype("int")

                    # Use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))

                    # Update list of bounding box coordinates, confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

        # Ensure at least one detection exists
        if len(indices) > 0:

            # Loop over the indexes we are keeping
            for i in indices.flatten():

                # Extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # Draw a bounding box rectangle and label on the image
                color = [int(c) for c in self.COLORS[class_ids[i]]]
                cv.rectangle(self.image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.labels[class_ids[i]], confidences[i])
                cv.putText(self.image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return self.image

# ------------------------------------------------------------------------------------------------------------------------------ #

# ---------------------------------------------- Main -------------------------------------------------------------------------- #


# Load our input image and grab its spatial dimensions
cap = cv.VideoCapture(0)

# Load labels
labels_path = '../../datasets/models/coco.names'
labels_stream = open(labels_path).read().strip().split("\n")

weights = '../../datasets/models/yolov3/yolov3-tiny.weights'
config = '../../datasets/models/yolov3/yolov3-tiny.cfg'

neuralnet = cv.dnn.readNetFromDarknet(config, weights)

det = Detector(neuralnet, labels_stream)


while True:

    # Capture frame by frame
    _, frame = cap.read()
    # Perform image processing


    # Perform object detection
    frame = det.detect(frame)

    # Display
    cv.imshow('Camera', frame)

    # Wait for ESC
    if cv.waitKey(1) == 27:
        break

# Print source resolution
print(det.image.shape)

cap.release()
cv.destroyAllWindows
