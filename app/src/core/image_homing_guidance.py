import cv2 as cv
import numpy as np
import app.src.unautil.utils as ut
from threading import Thread
import time

# TODO: ADD COMMENTS AND DOCSTRINGS
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

# ------------------------------------------------------------------------------------------------------------------------------ #

# -------------------------------------------------- Identifier ---------------------------------------------------------------- #


class Identifier:
    def __init__(self, target=None, n_features=100, hessian_thresh=100):
        self.target = target
        self.uav_frame = None
        self.n_features = n_features
        self.hessian_thresh = hessian_thresh
        self.reference_point = (0, 0)


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
        orb = cv.ORB_create(self.n_features, 1.1, 10, 2)
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
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        # Perform the matching between the ORB descriptors of the training image and the test image
        if target_descriptors.any() and uav_descriptors.any():
            matches = bf.match(target_descriptors, uav_descriptors)

            # The matches with shorter distance are the ones we want.
            matches = sorted(matches, key=lambda x: x.distance)

            return target_keypoints, uav_keypoints, matches
        return False, False, False


    def target_lock(self, uav_image, method):
        """
        Determines which of the current detections is the target
        Args:
            uav_image: Image from video feed
            method: Method used to extract features

        Returns: Index of target detection
        """
        target_keypoints, uav_keypoints, matches = self._compare_features(self.target, uav_image, method)

        if not matches:
            return False

        # Add target keypoints into a list
        matched_keypoints = [uav_keypoints[match.trainIdx] for match in matches]  # Append the corresponding UAV keypoint object

        # Clustering algorithm
        clusters = ut.fpnn(matched_keypoints, self.reference_point)

        clusters.sort(key=len)
        target_cluster = clusters[-1]

        uav_keyp_img = uav_image.copy()
        color = (0, 0, 0)
        for cluster in clusters:
            if cluster != target_cluster:
                while color == (0, 0, 255) or color == (0, 0, 0):
                    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                uav_keyp_img = cv.drawKeypoints(uav_keyp_img, cluster, outImage=None, color=(255, 255, 255))
            else:
                uav_keyp_img = cv.drawKeypoints(uav_keyp_img, cluster, outImage=None, color=(0, 0, 255))

        cv.imshow('UAV keypoints', uav_keyp_img)
        # cv.imshow('Target Matches', cv.drawMatches(self.target.image, target_keypoints, uav_image, uav_keypoints, matches, outImg=None))
        # Check which of the detection ROI's has the largest amount of keypoints associated with the target
        centroid_x, centroid_y = 0, 0
        for point in target_cluster:
            centroid_x, centroid_y = centroid_x + int(point.pt[0]), centroid_y + int(point.pt[1])
        if len(target_cluster):
            centroid = int(centroid_x / len(target_cluster)), int(centroid_y / len(target_cluster))
            self.reference_point = centroid
            return centroid
        return False


# ------------------------------------------------------------------------------------------------------------------------------ #

# ----------------------------------------------- User Interface --------------------------------------------------------------- #

class HomingUI(ut.UI):

    def __init__(self):
        super(HomingUI, self).__init__()
        self.bounding_box = []
        self.clicked = False

    def _get_mouse_coord(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN and not self.clicked:
            self.bounding_box.append((x, y))
            self.clicked = True
        elif event == cv.EVENT_LBUTTONDOWN and self.clicked:
            print('Target captured. Press ESC')
            self.bounding_box.append((x, y))

    def set_up_target(self, cam_cap):
        while True:
            _, feed = cam_cap.read()
            cv.imshow('Target Selection', feed)
            # Wait for ESC
            if cv.waitKey(1) & 0XFF == 27:
                cv.imwrite('../../datasets/testing/target.jpg', feed)
                cv.destroyAllWindows()
                break

        cv.namedWindow('Select ROI')
        cv.setMouseCallback('Select ROI', self._get_mouse_coord)
        while True:
            # Select target ROI
            cv.imshow('Select ROI', cv.imread('../../datasets/testing/target.jpg'))
            if cv.waitKey(1) == 27:
                cv.destroyAllWindows()
                break

        x, y = self.bounding_box[0]
        w, h = self.bounding_box[1]
        w, h = w - x, h - y
        return x, y, w, h

# ------------------------------------------------------------------------------------------------------------------------------ #


class ThreadedCamera(object):
    def __init__(self, src=0):
        self.cap = cv.VideoCapture(src)
        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 2)

        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1 / 30
        self.FPS_MS = int(self.FPS * 1000)

        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.cap.isOpened():
                (self.status, self.frame) = self.cap.read()
            time.sleep(self.FPS)

    def show_frame(self, frame=None):
        if frame is not None:
            cv.imshow('frame', frame)
            if cv.waitKey(self.FPS_MS) & 0XFF == 27:
                self.cap.release()
                cv.destroyAllWindows()
                exit()
        else:
            cv.imshow('frame', self.frame)
            if cv.waitKey(self.FPS_MS) & 0XFF == 27:
                self.cap.release()
                cv.destroyAllWindows()
                exit()


# ----------------------------------------------------- Main ------------------------------------------------------------------- #

cam_URL = 'http://192.168.2.12:8080/video'
cap = None

camera_index = 0
for i in range(1, 10):
    cap = cv.VideoCapture(i)
    if cap.isOpened():
        camera_index = i
        break

ui = HomingUI()

target_box = ui.set_up_target(cap)
target_centroid = (target_box[0] + target_box[2]) / 2 + (target_box[1] + target_box[3]) / 2
_, target_frame = cap.read()
target_frame = target_frame[target_box[1]:target_box[1]+target_box[3], target_box[0]: target_box[0] + target_box[2]]
cv.imshow('Cropped', target_frame)
cv.waitKey(0)
cap.release()

threaded_cam = ThreadedCamera(camera_index)
ident = Identifier(Target(target_box, target_centroid, target_frame), 50000)

while True:
    start = time.time()

    # Perform object detection
    try:
        res = ident.target_lock(threaded_cam.frame, 'ORB')
        if res:
            res_frame = ut.draw_image(threaded_cam.frame, res[0], res[1], 5, (0, 0, 255))
        else:
            res_frame = threaded_cam.frame
        # Display result
        threaded_cam.show_frame(res_frame)
        print('Operations took {:2f}s'.format(time.time() - start))

    except AttributeError:
        pass
