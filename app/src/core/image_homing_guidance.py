import cv2 as cv
import numpy as np
import time
import app.src.unautil.utils as ut


# --------------------------------------- Image Detection ---------------------------------------------------------------------- #
class Target:
    def __init__(self, target_id, category):
        self.target_id = target_id
        self.category = category
        self.images = []
        self.centroid = None

    def update_knowledge_base(self, img, center):
        if len(self.images) > 10:
            self.images.pop(0)
            self.images.append(img)
            self.centroid = center
            return
        self.images.append(img)
        self.centroid = center

    def verify_id(self, template):
        max_vals = []
        for img in self.images:
            res = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            max_vals.append(max_val)
            print(max_val)
        if any(x > 0.4 for x in max_vals):
            return True
        return False


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

        self.targets = []
        self.tid = 0

        self.start_time = time.time()


    def _init_network(self, model=None):
        if '.cfg' in self.config and '.weights' in self.weights:
            return cv.dnn.readNetFromDarknet(self.config, self.weights)


    def _extract_output_data(self, layer_outputs, img):
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

        img_height, img_width = img.shape[:2]

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
        Filters out overlapping bounding boxes and draws the rest of them on the image
        Args:


        Returns:

        """
        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = cv.dnn.NMSBoxes(self.boxes, self.confidences, 0.5, 0.2)
        centers_unsuppressed = self.centers.copy()
        self.centers = []
        # Ensure at least one detection exists
        if len(indices) > 0:

            # Loop over the indexes we are keeping
            for i in indices.flatten():
                # Extract the bounding box coordinates
                (x, y) = (self.boxes[i][0], self.boxes[i][1])
                (w, h) = (self.boxes[i][2], self.boxes[i][3])
                self.centers.append(centers_unsuppressed[i])

                # Assign an ID to the target using template matching on a list of target images
                tID = self._assign_id(i, self._capture_target(i), centers_unsuppressed[i])

                print(tID)

                # Draw a bounding box rectangle and label on the image
                color = [int(c) for c in self.COLORS[self.class_ids[i]]]
                cv.rectangle(self.image, (x, y), (x + w, y + h), color, 2)
                text = "{} : {:.4f}".format(self.labels[self.class_ids[i]], self.confidences[i])
                cv.putText(self.image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                ut.draw_image(self.image, centers_unsuppressed[i][0], centers_unsuppressed[i][1], 5, color)

    def _capture_target(self, i):
        # print(self.boxes[i])
        target_image = ut.snap_image(self.image, self.boxes[i][0], self.boxes[i][1], width=self.boxes[i][2], height=self.boxes[i][3])
        # cv.imshow('Target', target_image)
        return target_image

    def _assign_id(self, i, img, centers):
        """
        Does something wrong
        Returns:

        """

        if not self.targets:
            target = Target(self.tid, self.class_ids[i])
            target.update_knowledge_base(img, centers)
            self.targets.append(target)
            return target.target_id
        else:
            # Check if target is already in the target list
            for target in self.targets:
                # If the target is in a 100 pixel radius then, it's not a new object
                if np.abs(centers[0] - target.centroid[0]) < 100 and np.abs(centers[1] - target.centroid[1]) < 100:
                    target.update_knowledge_base(image, centers)
                    return target.target_id
                # Apply template matching to see if the current target exists in the knowledge base
                elif not target.verify_id(img):
                    print("Not the same object")
                    if len(self.targets) > 3:
                        self.targets.pop(0)
                    self.tid += 1
                    new_target = Target(self.tid, self.class_ids[i])
                    new_target.update_knowledge_base(img, centers)
                    self.targets.append(new_target)
                    return new_target.target_id

                else:
                    target.update_knowledge_base(img, centers)
                    return target.target_id


    def detect(self, img, blob_size=(320, 320)):

        """

        Args:
            img: Image in which object detection will perform on
            blob_size: Shape of blob used in dnn.blobFromImage
        Returns:

        """
        # Initialize image
        self.image = img

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

    def __init__(self):
        print('Guide')


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

    image = det.detect(frame, blob_size=(128, 128))

    # Perform detection and display result
    try:
        cv.imshow('Camera', image)
        ui.mouse_select()
    except cv.error:
        cv.destroyAllWindows()

    # Wait for ESC
    if cv.waitKey(1) == 27:
        break


cap.release()
cv.destroyAllWindows()
