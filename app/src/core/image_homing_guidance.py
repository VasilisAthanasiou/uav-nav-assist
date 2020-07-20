import cv2 as cv
import numpy as np
import time
import app.src.unautil.utils as ut

# TODO: Make target identification work for multiple objects at the same time.


# -------------------------------------------------- Target Classes --------------------------------------------------------------- #

class Targets:
    def __init__(self, max_objects):
        self.target_list = []
        self.max_objects = max_objects

    def return_index(self, target_id):
        """
        Uses the target id to return the corresponding index on all the Targets fields
        Args:
            target_id: Unique object ID

        Returns: Index of object with ID = target_id

        """
        return [target.target_id for target in self.target_list].index(target_id)

    def insert_target(self, target, target_id):
        """
        Args:
            target: Uninitialized Target object
            target_id: Unique object ID
        """
        new_target = Target(target_id, target.max_images)
        new_target.update_data(target.images, target.centroid)

        if len(self.target_list) >= self.max_objects:
            self.pop_target()
            self.target_list.append(new_target)
            return
        self.target_list.append(new_target)

    def pop_target(self):
        """
        Removes a target from the target_list
        """
        self.target_list.pop()

    def verify_id(self, method):
        print('verify id')



class Target:
    def __init__(self, target_id=-1, image=None):
        self.target_id = target_id
        self.image = image
        self.centroid = None

    def initialize_target(self, target_id):
        self.target_id = target_id

    def update_data(self, centroid, image):
        self.image = image
        self.centroid = centroid

    def reset_target(self):
        self.__init__()

# ------------------------------------------------------------------------------------------------------------------------------ #

# -------------------------------------------------- Identifier ---------------------------------------------------------------- #

class Identifier:
    def __init__(self, n_features=100, hessian_thresh=100):
        self.candidate_target = Target()
        self.n_features = n_features
        self.hessian_thresh = hessian_thresh
        self.tid = 0

    def _use_surf(self, image):
        surf = cv.xfeatures2d_SURF()
        surf.create(self.hessian_thresh)
        return surf.detect(image)

    def _use_orb(self, image):
        orb = cv.ORB_create(self.n_features)
        keypoints = orb.detect(image)
        return orb.compute(image, keypoints)

    def _extract_features(self, method, image):
        if method == 'SURF':  # Note that SURF in not free
            return self._use_surf(image)
        elif method == 'ORB':
            return self._use_orb(image)
        else:
            print('No valid method selected')

    def _compare_features(self, target_image, uav_image, method):
        target_keypoints, target_descriptors = self._extract_features(method, target_image)
        uav_keypoints, uav_descriptors = self._extract_features(method, uav_image)
        target_keyp_img = cv.drawKeypoints(target_image, target_keypoints, outImage=None)
        uav_keyp_img = cv.drawKeypoints(uav_image, uav_keypoints, outImage=None)
        cv.imshow('Target keypoints', target_keyp_img)
        cv.imshow('UAV keypoints', uav_keyp_img)


    def target_lock(self, target_image, uav_image, method):
        self._compare_features(target_image, uav_image, method)

    def assign_id(self, image, centroid):
        """
        Programming horror. Proceed with caution...
        Returns: ID

        """
        print('Assign ID')

# ------------------------------------------------------------------------------------------------------------------------------ #

# -------------------------------------------------- Detector ------------------------------------------------------------------ #

class Detector:
    def __init__(self, config, weights, labels):
        """
        THIS CLASS HAS TOO MANY FIELDS AND METHODS! SHOULD BE BROKEN INTO TWO DIFFERENT CLASSES!
        Args:
            config: Path for the network configuration file
            weights: Path for the network weights file
            labels: Path for the network labels file
        """

        # ----------------------------------- Network fields ---------------------------------------------------- #
        self.config, self.weights = config, weights
        self.image = None
        self.prev_frame = None
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
        # ------------------------------------------------------------------------------------------------------- #

        self.identifier = Identifier()


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

                # Update the centroid
                self.prev_centroid = centers_unsuppressed[i]

                # Draw features
                if self.prev_frame is not None:
                    self.identifier.target_lock(self.prev_frame, self.image, 'ORB')

                # Draw a bounding box rectangle and label on the image
                color = [int(c) for c in self.COLORS[self.class_ids[i]]]
                cv.rectangle(self.image, (x, y), (x + w, y + h), color, 2)
                text = "{} {:.4f}".format(self.labels[self.class_ids[i]], self.confidences[i])
                cv.putText(self.image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                ut.draw_image(self.image, centers_unsuppressed[i][0], centers_unsuppressed[i][1], 5, color)
                self.prev_frame = self.image

    def _capture_target(self, i):
        """
        Capture an image of a detected object
        Args:
            i: Index

        Returns: Cropped image
        """
        target_image = ut.snap_image(self.image, self.boxes[i][0], self.boxes[i][1], width=self.boxes[i][2], height=self.boxes[i][3])
        # cv.imshow('Target', target_image)
        return target_image


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
    ret, frame = cap.read()
    # Perform image processing

    # Perform object detection
    det_frame = det.detect(frame, blob_size=(190, 190))

    # Display result
    if ret:
        cv.imshow('Camera', det_frame)
    # cv.waitKey(0)

    # Wait for ESC
    if cv.waitKey(1) & 0XFF == 27:
        break

# ident = Identifier(300)
# target = cv.imread('../../datasets/testing/target.jpg')
# uav = cv.imread('../../datasets/testing/uav.jpg')
#
# ident.target_lock(target, uav, 'ORB')
#
#

# cap.release()
cv.destroyAllWindows()
