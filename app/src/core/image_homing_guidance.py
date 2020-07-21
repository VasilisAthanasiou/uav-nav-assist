import cv2 as cv
import numpy as np
import app.src.unautil.utils as ut


# -------------------------------------------------- Target Classes --------------------------------------------------------------- #

class Target:
    def __init__(self, bounding_box, centroid, image=None, target_id=0):
        self.target_id = target_id
        self.image = image
        self.bounding_box = bounding_box
        self.centroid = centroid

    def update_data(self, bounding_box, centroid):
        self.bounding_box = bounding_box
        self.centroid = centroid

    def reset_target(self):
        self.__init__()

# ------------------------------------------------------------------------------------------------------------------------------ #

# -------------------------------------------------- Identifier ---------------------------------------------------------------- #


class Identifier:
    def __init__(self, target=None, n_features=100, hessian_thresh=100):
        self.target = target
        self.boxes = None
        self.n_features = n_features
        self.hessian_thresh = hessian_thresh
        self.tid = 0

    def _use_surf(self, image):
        """
        Use Speeded Up Robust Features
        Args:
            image: Image that SURF will be applied to.
        Returns: Keypoints and descriptors
        """
        surf = cv.xfeatures2d_SURF()
        surf.create(self.hessian_thresh)
        return surf.detect(image)

    def _use_orb(self, image):
        """
        Use Oriented FAST and Rotated BRIEF
        Args:
            image: Image that ORB will be applied to.
        Returns: Keypoints and descriptors
        """
        orb = cv.ORB_create(self.n_features)
        keypoints = orb.detect(image)
        return orb.compute(image, keypoints)

    def _extract_features(self, method, image):
        """
        Calls a method for feature extraction
        Args:
            method: Method that will be called
            image: Image that the method will extract features from

        Returns: Keypoints and Descriptors

        """
        if method == 'SURF':  # Note that SURF in not free
            return self._use_surf(image)
        elif method == 'ORB':
            return self._use_orb(image)
        else:
            print('No valid method selected')

    def _compare_features(self, target, uav_image, method):
        """
        Matches features from a pre-determined target with a UAV image
        Args:
            target: Target object
            uav_image: Image from video feed
            method: Method that will be used to extract features

        Returns: Target keypoints, UAV keypoints and a match object that associates them
        """
        target_keypoints, target_descriptors = self._extract_features(method, target.image)
        uav_keypoints, uav_descriptors = self._extract_features(method, uav_image)
        target_keyp_img = cv.drawKeypoints(target.image, target_keypoints, outImage=None)
        uav_keyp_img = cv.drawKeypoints(uav_image, uav_keypoints, outImage=None)
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        # Perform the matching between the ORB descriptors of the training image and the test image
        matches = bf.match(target_descriptors, uav_descriptors)

        # The matches with shorter distance are the ones we want.
        matches = sorted(matches, key=lambda x: x.distance)

        result = cv.drawMatches(target_keyp_img, target_keypoints, uav_keyp_img, uav_keypoints, matches, uav_image, flags=2)
        # cv.imshow('Target keypoints', result)
        return target_keypoints, uav_keypoints, matches

    def set_target(self, target, boxes):
        """
        Sets a target up
        Args:
            target: Target object
            boxes: Bounding boxes from detected objects
        """
        self.target = target
        self.boxes = boxes

    def target_lock(self, uav_image, method):
        """
        Determines which of the current detections is the target
        Args:
            uav_image: Image from video feed
            method: Method used to extract features

        Returns: Index of target detection
        """
        target_keypoints, uav_keypoints, matches = self._compare_features(self.target, uav_image, method)
        (tx, ty) = self.target.bounding_box[0], self.target.bounding_box[1]
        (tw, th) = self.target.bounding_box[2], self.target.bounding_box[3]

        # Add that target keypoints into a list
        target_matches = []
        for match in matches:
            tkeyp_x, tkeyp_y = target_keypoints[match.queryIdx].pt  # queryIdx is the index of a target keypoint that got matched
            if tx + tw >= int(tkeyp_x) >= tx and ty + th >= int(tkeyp_y) >= ty:  # If the matched keypoint is within the target ROI
                target_matches.append(uav_keypoints[match.trainIdx].pt)  # Append the corresponding UAV keypoint

        box_score = []
        # Check which of the detection ROI's has the largest amount of keypoints associated with the target
        for i in range(len(self.boxes)):
            (bx, by) = self.boxes[i][0], self.boxes[i][1]
            (bw, bh) = self.boxes[i][2], self.boxes[i][3]
            score = 0
            for tmatch in target_matches:
                if bx + bw >= int(tmatch[0]) >= bx and by + bh >= int(tmatch[1]) >= by:
                    score += 1
                    target_matches.remove(tmatch)
            box_score.append(score)

        if max(box_score) > 25 and len(self.boxes) > 1:
            return box_score.index(max(box_score))
        return -1


# ------------------------------------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------ Detector -------------------------------------------------------------- #

class Detector:
    def __init__(self, config, weights, labels):
        """
        Args:
            config: Path for the network configuration file
            weights: Path for the network weights file
            labels: Path for the network labels file
        """

        # ----------------------------------- Network fields ---------------------------------------------------- #
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
        # ------------------------------------------------------------------------------------------------------- #
        self.target = Target([319, 182, 307, 244], ((319+307)/2, (182+244)/2), image=cv.imread('../../datasets/testing/target.jpg'))
        self.target_update_counter = 0
        self.prev_index = -1
        self.identifier = Identifier(n_features=1000)


    def _init_network(self):
        """
        Initializes Neural Network pre-trained model
        """
        if '.cfg' in self.config and '.weights' in self.weights:
            return cv.dnn.readNetFromDarknet(self.config, self.weights)

    def _extract_output_data(self, layer_outputs, img):
        """Takes in network output data and processes it into useful data that will be used
        for bounding boxes, confidences and class IDs
        Args:
            layer_outputs: Output data produced from net.forward()
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
        """
        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = cv.dnn.NMSBoxes(self.boxes, self.confidences, 0.5, 0.2)
        centers_unsuppressed = self.centers
        self.centers = []
        res_image = self.image.copy()
        target_index = -1

        # Initialize target
        self.identifier.set_target(self.target, self.boxes)

        # Lock target
        if len(self.boxes) != 0:
            target_index = self.identifier.target_lock(res_image, 'ORB')

        # Ensure at least one detection exists
        if len(indices) > 0:

            # Loop over the indexes we are keeping
            for i in indices.flatten():
                # Extract the bounding box coordinates
                (x, y) = (self.boxes[i][0], self.boxes[i][1])
                (w, h) = (self.boxes[i][2], self.boxes[i][3])
                self.centers.append(centers_unsuppressed[i])

                if i == target_index:
                    tcenter_x, tcenter_y = centers_unsuppressed[i]
                    cv.rectangle(self.image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    text = "{} {:.4f}".format(self.labels[self.class_ids[i]], self.confidences[i])
                    cv.putText(self.image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    ut.draw_image(self.image, centers_unsuppressed[i][0], centers_unsuppressed[i][1], 5, (0, 0, 255))
                    self.target_update_counter += 1
                    print('Target update expected in: {} frames'.format(10 - self.target_update_counter))
                    if self.target_update_counter >= 10 and self.prev_index == i:
                        print('Updated target image')
                        self.target.image = self.image  # TODO: Make a function for this
                        self.target.update_data(self.boxes[i], (tcenter_x, tcenter_y))
                        self.target_update_counter = 0
                    self.prev_index = i
                else:
                    # Draw a bounding box rectangle and label on the image
                    cv.rectangle(self.image, (x, y), (x + w, y + h), (255, 255, 255), 2)
                    text = "{} {:.4f}".format(self.labels[self.class_ids[i]], self.confidences[i])
                    cv.putText(self.image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    ut.draw_image(self.image, centers_unsuppressed[i][0], centers_unsuppressed[i][1], 5, (255, 255, 255))


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
        Returns: Drawn image

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

# ----------------------------------------------- Homing Guidance -------------------------------------------------------------- #

class Guide:

    def __init__(self):
        print('Guide')


# ------------------------------------------------------------------------------------------------------------------------------ #

# ----------------------------------------------- User Interface --------------------------------------------------------------- #

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

    def set_up_target(self, cap):
        while True:
            _, feed = cap.read()
            cv.imshow('Target Selection', feed)
            # Wait for ESC
            if cv.waitKey(1) & 0XFF == 27:
                cv.imwrite('../../datasets/testing/target.jpg', feed)
                break





# ------------------------------------------------------------------------------------------------------------------------------ #

# ----------------------------------------------------- Main ------------------------------------------------------------------- #

# Load labels
labels_path = '../../datasets/models/coco.names'
labels_stream = open(labels_path).read().strip().split("\n")

config_path = '../../datasets/models/yolov3/yolov3-tiny.cfg'
weights_path = '../../datasets/models/yolov3/yolov3-tiny.weights'

det = Detector(config_path, weights_path, labels_stream)
ui = HomingUI()

# Load our input image and grab its spatial dimensions
cap = cv.VideoCapture(0)

ui.set_up_target(cap)

while True:

    # Capture frame by frame
    ret, frame = cap.read()
    # Perform image processing

    # Perform object detection
    det_frame = det.detect(frame, blob_size=(320, 320))

    # Display result
    if ret:
        cv.imshow('Camera', det_frame)

    # Wait for ESC
    if cv.waitKey(1) & 0XFF == 27:
        break

# cap.release()
cv.destroyAllWindows()
