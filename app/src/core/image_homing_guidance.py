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

        self.boxes = []
        self.confidences = []
        self.class_ids = []
        self.centers = []

        self.start_time = time.time()


    def _init_network(self, model=None):
        if '.cfg' in self.config and '.weights' in self.weights:
            return cv.dnn.readNetFromDarknet(self.config, self.weights)


    def _extract_output_data(self, layer_outputs, image):
        """Takes in network output data and processes it into useful data that will be used
        for bounding boxes, confidences and class IDs

        Args:
            layer_outputs: Output data produced from net.forward()

        Returns:

        """
        # Initialize lists of detected bounding boxes, confidences, and class IDs
        self.boxes = []
        self.confidences = []
        self.class_ids = []
        self.centers = []

        img_height, img_width = image.shape[:2]

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

                    box = detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                    (center_x, center_y, det_width, det_height) = box.astype("int")

                    # Use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(center_x - (det_width / 2))
                    y = int(center_y - (det_height / 2))

                    # Update list of bounding box coordinates, confidences, and class IDs
                    self.boxes.append([x, y, int(det_width), int(det_height)])
                    self.confidences.append(float(confidence))
                    self.class_ids.append(class_id)
                    self.centers.append((center_x, center_y))

    def _draw_boxes(self):
        """

        Args:
            boxes: List of top left and bottom right coordinates for bounding boxes (top_x, top_y , bot_x, bot_y)
            confidences: List of probabilities of possible class for sensed object
            class_ids: List of class IDs for most probable classes

        Returns:

        """
        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = cv.dnn.NMSBoxes(self.boxes, self.confidences, 0.5, 0.3)

        # Ensure at least one detection exists
        if len(indices) > 0:

            # Loop over the indexes we are keeping
            for i in indices.flatten():

                # Extract the bounding box coordinates
                (x, y) = (self.boxes[i][0], self.boxes[i][1])
                (w, h) = (self.boxes[i][2], self.boxes[i][3])

                # Draw a bounding box rectangle and label on the image
                color = [int(c) for c in self.COLORS[self.class_ids[i]]]
                cv.rectangle(self.image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.labels[self.class_ids[i]], self.confidences[i])
                cv.putText(self.image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                ut.draw_image(self.image, self.centers[i][0], self.centers[i][1], 5, color)


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
        self._extract_output_data(layer_outputs, self.image)

        # Draw boxes
        self._draw_boxes()

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

# Load labels
labels_path = '../../datasets/models/coco.names'
labels_stream = open(labels_path).read().strip().split("\n")

config_path = '../../datasets/models/yolov3/yolov3-tiny.cfg'
weights_path = '../../datasets/models/yolov3/yolov3-tiny.weights'

det = Detector(config_path, weights_path, labels_stream)
ui = HomingUI()

# Load our input image and grab its spatial dimensions
cap = cv.VideoCapture(0)

while True:

    # Capture frame by frame
    _, frame = cap.read()
    # Perform image processing

    # Perform object detection


    # Display
    # try:
    cv.imshow('Camera', det.detect(frame, blob_size=(320, 320)))
    # ui.mouse_select()
    # except cv.error:
    #     cv.destroyAllWindows()

    print(4)
    # Wait for ESC
    if cv.waitKey(1) == 27:
        break

# Print source resolution
print(det.image.shape)

cap.release()
cv.destroyAllWindows()
#
#
# # Load labels
# labels_path = '../../datasets/models/coco.names'
# LABELS = open(labels_path).read().strip().split("\n")
#
# config_path = '../../datasets/models/yolov3/yolov3-tiny.cfg'
# weights_path = '../../datasets/models/yolov3/yolov3-tiny.weights'
#
# np.random.seed(42)
# COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
#
# # load our YOLO object detector trained on COCO dataset (80 classes)
# print("[INFO] loading YOLO from disk...")
# net = cv.dnn.readNetFromDarknet(config_path, weights_path)
#
# # determine only the *output* layer names that we need from YOLO
# ln = net.getLayerNames()
# ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#
# cap = cv.VideoCapture(0)
#
#
# while True:
#     ret, frame = cap.read()
#     # load our input image and grab its spatial dimensions
#     image = frame
#     (H, W) = image.shape[:2]
#     # construct a blob from the input image and then perform a forward
#     # pass of the YOLO object detector, giving us our bounding boxes and
#     # associated probabilities
#     blob = cv.dnn.blobFromImage(image, 1 / 255.0, (320, 320), swapRB=True, crop=False)
#     net.setInput(blob)
#     start = time.time()
#     layerOutputs = net.forward(ln)
#     end = time.time()
#     # show timing information on YOLO
#     print("[INFO] YOLO took {:.6f} seconds".format(end - start))
#
#     boxes = []
#     confidences = []
#     classIDs = []
#
#     # loop over each of the layer outputs
#     for output in layerOutputs:
#         # loop over each of the detections
#         for detection in output:
#             # extract the class ID and confidence (i.e., probability) of
#             # the current object detection
#             scores = detection[5:]
#             classID = np.argmax(scores)
#             confidence = scores[classID]
#             # filter out weak predictions by ensuring the detected
#             # probability is greater than the minimum probability
#             if confidence > 0.5:
#                 # scale the bounding box coordinates back relative to the
#                 # size of the image, keeping in mind that YOLO actually
#                 # returns the center (x, y)-coordinates of the bounding
#                 # box followed by the boxes' width and height
#                 box = detection[0:4] * np.array([W, H, W, H])
#                 (centerX, centerY, width, height) = box.astype("int")
#                 # use the center (x, y)-coordinates to derive the top and
#                 # and left corner of the bounding box
#                 x = int(centerX - (width / 2))
#                 y = int(centerY - (height / 2))
#                 # update our list of bounding box coordinates, confidences,
#                 # and class IDs
#                 boxes.append([x, y, int(width), int(height)])
#                 confidences.append(float(confidence))
#                 classIDs.append(classID)
#
#     idxs = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
#
#     # ensure at least one detection exists
#     if len(idxs) > 0:
#         # loop over the indexes we are keeping
#         for i in idxs.flatten():
#             # extract the bounding box coordinates
#             (x, y) = (boxes[i][0], boxes[i][1])
#             (w, h) = (boxes[i][2], boxes[i][3])
#             # draw a bounding box rectangle and label on the image
#             color = [int(c) for c in COLORS[classIDs[i]]]
#             cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
#             text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
#             cv.putText(image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#     # show the output image
#     cv.imshow("Image", image)
#     if cv.waitKey(1) == 27:
#         break

















