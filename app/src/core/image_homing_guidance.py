import cv2 as cv
import numpy as np
import os
import imutils
import time
import app.src.unautil.utils as ut

# --------------------------------------- Image Detection ---------------------------------------------------------------------- #

class Detector:
    def __init__(self, config, weights, labels, blob_size=(320, 320)):
        """

        Args:
            config: Path for the network configuration file
            weights: Path for the network weights file
            labels: Path for the network labels file
        """
        self.config, self.weights = config, weights
        self.image = None
        self.labels = labels

        # Create a list of colors that will be used for bounding boxes
        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")


        # Initialize network
        self.net = self._init_network()
        # Determine the *output* layer names needed from YOLO
        self.layer_names = [self.net.getLayerNames()[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        self.start_time = time.time()


    def _init_network(self, model=None):
        if '.cfg' in self.config and '.weights' in self.weights:
            return cv.dnn.readNetFromDarknet(self.config, self.weights)


    def _extract_output_data(self, layer_outputs):
        """Takes in network output data and processes it into useful data that will be used
        for bounding boxes, confidences and class IDs

        Args:
            layer_outputs: Output data produced from net.forward()

        Returns:

        """
        # Initialize lists of detected bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []
        centers = []
        height, width = self.image.shape[:2]

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
                    boxes.append([x, y, x + int(width), y + int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    centers.append((center_x, center_y))

        return boxes, confidences, class_ids, centers


    def _draw_boxes(self, boxes, confidences, class_ids, centers):
        """

        Args:
            boxes: List of top left and bottom right coordinates for bounding boxes (top_x, top_y , bot_x, bot_y)
            confidences: List of probabilities of possible class for sensed object
            class_ids: List of class IDs for most probable classes

        Returns:

        """
        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

        # Ensure at least one detection exists
        if len(indices) > 0:

            # Loop over the indexes we are keeping
            for i in indices.flatten():

                # Extract the bounding box coordinates
                (x_start, y_start) = (boxes[i][0], boxes[i][1])
                (x_end, y_end) = (boxes[i][2], boxes[i][3])

                # Draw a bounding box rectangle and label on the image
                color = [int(c) for c in self.COLORS[class_ids[i]]]
                cv.rectangle(self.image, (x_start, y_start), (x_end, y_end), color, 2)
                text = "{}: {:.4f}".format(self.labels[class_ids[i]], confidences[i])
                cv.putText(self.image, text, (x_start, y_start - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                ut.draw_image(self.image, centers[i][0], centers[i][1], 5, color)


    def detect(self, image, blob_size=(320, 320)):

        """

        Args:
            image: Image in which object detection will perform on
            blob_size: Shape of blob used in dnn.blobFromImage
        Returns:

        """
        # Initialize image
        self.image = image

        # Construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv.dnn.blobFromImage(self.image, 1 / 255.0, blob_size, swapRB=True, crop=False)
        self.net.setInput(blob)

        layer_outputs = self.net.forward(self.layer_names)

        # Process the output layer data
        boxes, confidences, class_ids, centers = self._extract_output_data(layer_outputs)

        self._draw_boxes(boxes, confidences, class_ids, centers)

        return self.image

# ------------------------------------------------------------------------------------------------------------------------------ #

# ------------------------------------------ Homing Guidance ------------------------------------------------------------------- #

class Guide:

    def course_correct(self):
        print('Course correct')


# ------------------------------------------------------------------------------------------------------------------------------ #

# ------------------------------------------ User Interface -------------------------------------------------------------------- #

class HomingUI(ut.UI):

    def __init__(self):
        super(HomingUI, self).__init__()
        self.mouse_x, self.mouse_y = None, None

    def mouse_select(self):
        return cv.setMouseCallback('Camera', self._get_mouse_coord)

    def _get_mouse_coord(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.mouse_x, self.mouse_y = x, y
            mouse_loc = (self.mouse_x, self.mouse_y)
            print(mouse_loc)


# ------------------------------------------------------------------------------------------------------------------------------ #

# ---------------------------------------------- Main -------------------------------------------------------------------------- #

# Load our input image and grab its spatial dimensions
cap = cv.VideoCapture(0)

# Load labels
labels_path = '../../datasets/models/coco.names'
labels_stream = open(labels_path).read().strip().split("\n")

config_path = '../../datasets/models/yolov3/yolov3-tiny.cfg'
weights_path = '../../datasets/models/yolov3/yolov3-tiny.weights'

det = Detector(config_path, weights_path, labels_stream)
ui = HomingUI()

while True:

    # Capture frame by frame
    _, frame = cap.read()
    # Perform image processing


    # Perform object detection
    frame = det.detect(frame)

    # Display
    cv.imshow('Camera', frame)
    ui.mouse_select()

    # Wait for ESC
    if cv.waitKey(1) == 27:
        break

# Print source resolution
print(det.image.shape)

cap.release()
cv.destroyAllWindows()

