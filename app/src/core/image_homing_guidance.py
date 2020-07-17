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
        new_target = Target(target_id, target.max_images, target.category)
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

    def _TM_VER_ID(self, template_target):
        """
        Use Template Matching to examine if the template is of a known object
        Args:
            template:

        Returns:

        """
        max_val = 0
        for target in self.target_list:
            print('Number of targets {}'.format(len(self.target_list)))
            if template_target.category == target.category:
                print('Number of target images {}'.format(len(target.images)))
                print('Number of template images {}'.format(len(template_target.images)))
                for target_img in target.images:
                    tar_width, tar_height, _ = target_img.shape
                    for template in template_target.images:

                        try:
                            template = cv.resize(template, (tar_height, tar_width))
                            if target_img.shape == template.shape:
                                start_time = time.time()
                                # Perform template matching
                                res = cv.matchTemplate(ut.process_image(target_img, resize=0.5), ut.process_image(template, resize=2), cv.TM_CCOEFF_NORMED)
                                min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

                                end_time = time.time()
                                print('Template Matching took {:.2f}s with {} correlation'.format(end_time - start_time, max_val))
                            else:
                                print(target_img.shape, template.shape)
                        except (cv.error, ZeroDivisionError) as e:
                            print(e)

                        if max_val > 0.5:
                            return True, target.target_id
        return False, -1

    def _TM_UPDATE_KB(self, target_id, template, center):
        target_index = self.return_index(target_id)
        self.target_list[target_index].update_data(template, center)

    def _use_template_matching(self, method, template=None, target_id=None, center=None, target=None):
        if method == 'update knowledge base':
            return self._TM_UPDATE_KB(target_id, template, center)
        if method == 'verify id':
            return self._TM_VER_ID(target)

    def verify_id(self, target, method):
        if method == 'template matching' and target:
            return self._use_template_matching('verify id', target=target)

    def update_knowledge_base(self, target_id, template, center, method):
        if method == 'template matching':
            return self._use_template_matching('update knowledge base', template, target_id, center)


class Target:
    def __init__(self, target_id=-1, max_images=120, category=None):
        self.target_id = target_id
        self.centroid = None
        self.category = category
        self.images = []
        self.max_images = max_images

    def initialize_target(self, target_id, category=None):
        self.target_id = target_id
        if category:
            self.category = category

    def update_data(self, img, centroid):
        self.centroid = centroid
        if type(img) is list:
            for i in img:
                self.images.append(i)
                return

        if len(self.images) >= self.max_images:
            self.images.pop()
            self.images.append(img)
            return
        self.images.append(img)

    def reset_target(self, max_images):
        self.__init__(max_images=max_images)

# ------------------------------------------------------------------------------------------------------------------------------ #

# -------------------------------------------------- Identifier ---------------------------------------------------------------- #

class Identifier:
    def __init__(self, labels):
        self.targets = Targets(5)
        self.candidate_target = Target(max_images=20)
        self.tid = 0
        self.prev_centroid = (0, 0)
        self.labels = labels

    def _are_same_class(self, label_index, tid):
        """
        Check if one detected object is the same class as a given target
        Args:
            i: Current object index
            tid: Target ID. Is used with Targets.return_index() to return the index of a given target

        Returns: Boolean
        """
        return self.labels[label_index] == [target.category for target in self.targets.target_list][self.targets.return_index(tid)]

    def assign_id(self, label_index, img, center):
        """
        Programming horror. Proceed with caution...
        Returns: ID

        """
        # Check if the candidate target is initialized
        if self.candidate_target.target_id == -1:
            self.candidate_target.initialize_target(self.tid, self.labels[label_index])

        same_loc_and_category = np.abs(center[0] - self.prev_centroid[0]) < 100 and np.abs(center[1] - self.prev_centroid[1]) < 100 and self.labels[label_index] == self.candidate_target.category

        # Before identifying targets, make sure there is adequate data on the target
        if len(self.candidate_target.images) >= self.candidate_target.max_images and not same_loc_and_category:

            if not self.targets.target_list:
                self.targets.insert_target(self.candidate_target, 0)
                return 0
            elif np.abs(center[0] - self.prev_centroid[0]) < 50 and np.abs(center[1] - self.prev_centroid[1]) < 50 and self._are_same_class(label_index, self.tid + 1):
                self.targets.update_knowledge_base(self.tid - 1, img, center, method='template matching')
                return self.tid - 1
            else:
                # Check if this is a new object
                matched, matched_id = self.targets.verify_id(self.candidate_target, method='template matching')
                print(matched_id)
                if matched and self._are_same_class(label_index, matched_id):
                    print('Matched')
                    self.targets.update_knowledge_base(matched_id, img, center, method='template matching')
                    return matched_id

            print('Initialized object')
            self.targets.insert_target(self.candidate_target, self.tid)
            self.tid += 1
            self.candidate_target.reset_target(20)
            self.candidate_target.update_data(img, center)
            self.prev_centroid = center
            return self.tid
        else:
            print(self.candidate_target.category)
            if same_loc_and_category:
                self.candidate_target.update_data(img, center)
            else:
                self.candidate_target.reset_target(20)
                self.candidate_target.update_data(img, center)
            self.prev_centroid = center

        return self.candidate_target.target_id

# ------------------------------------------------------------------------------------------------------------------------------ #

# -------------------------------------------------- Detector ------------------------------------------------------------------ #

class Detector:
    def __init__(self, config, weights, labels, blob_size=(320, 320)):
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
        self.identifier = Identifier(self.labels)


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
                tID = self.identifier.assign_id(i, self._capture_target(i), centers_unsuppressed[i])
                print('TID = {}, Class = {}'.format(tID, self.labels[self.class_ids[i]]))

                # Update the centroid
                self.prev_centroid = centers_unsuppressed[i]

                # Draw a bounding box rectangle and label on the image
                color = [int(c) for c in self.COLORS[self.class_ids[i]]]
                cv.rectangle(self.image, (x, y), (x + w, y + h), color, 2)
                text = "{} ID:{} : {:.4f}".format(self.labels[self.class_ids[i]], tID, self.confidences[i])
                cv.putText(self.image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                ut.draw_image(self.image, centers_unsuppressed[i][0], centers_unsuppressed[i][1], 5, color)

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
    det_frame = det.detect(frame, blob_size=(320, 320))

    # Display result
    if ret:
        cv.imshow('Camera', det_frame)
    # cv.waitKey(0)

    # Wait for ESC
    if cv.waitKey(1) & 0XFF == 27:
        break

cap.release()
cv.destroyAllWindows()
