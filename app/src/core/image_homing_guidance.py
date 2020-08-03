import cv2 as cv
import numpy as np
import app.src.unautil.utils as ut
from threading import Thread
import time

# -------------------------------------------------- Feature Extractor ---------------------------------------------------------------- #
class FeatureExtractor:
    def __init__(self, n_features=10000, hessian_thresh=100):
        self.n_features = n_features
        self.hessian_thresh = hessian_thresh

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
        orb = cv.ORB_create(self.n_features, 1.005, 20, 2)  # TODO: Read more about ORB and tweak parameters to better suit the problem
        keypoints = orb.detect(image)
        return orb.compute(image, keypoints)

    def extract_features(self, method, image):
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

    def match_features(self, target_descriptors, uav_image, method):
        """
        Matches features from a pre-determined target with a UAV image
        Args:
            target_descriptors: Descriptor of selected target features
            uav_image: Image from video feed
            method: Method that will be used to extract features
        Returns: Target keypoints, UAV keypoints and a match object that associates them
        """

        uav_keypoints, uav_descriptors = self.extract_features(method, uav_image)
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        # Perform the matching between the ORB descriptors of the training image and the test image
        if target_descriptors.any() and uav_descriptors.any():
            matches = bf.match(target_descriptors, uav_descriptors)

            return uav_keypoints, matches
        return False, False, False

# ------------------------------------------------------------------------------------------------------------------------------ #

# ----------------------------------------------- Tracker --------------------------------------------------------------- #

class Tracker:
    def __init__(self, target_image=None, n_features=10000, hessian_thresh=100, nn_dist=50):
        self.target_image = target_image
        self.uav_image = None
        self.feature_extractor = FeatureExtractor(n_features, hessian_thresh)
        self.reference_point = (0, 0)
        self.nn_dist = nn_dist
        self.target_keypoints, self.target_descriptors = None, None

    def track(self, uav_frame, method):
        self.uav_image = uav_frame
        uav_keypoints, matches = self.feature_extractor.match_features(self.target_descriptors, uav_frame, method)
        if not matches:
            return False
        return self._target_lock(self.target_keypoints, uav_keypoints, matches, self.nn_dist)

    def initialize_target(self, method, target_image):
        self.target_keypoints, self.target_descriptors = self.feature_extractor.extract_features(method, target_image)

    def _target_lock(self, target_keypoints, uav_keypoints, matches, nn_dist):
        """
        Determines which of the current detections is the target
        Args:
            matches:
            uav_keypoints:
            target_keypoints:
            nn_dist: Minimum distance of nearest neighbor
        Returns: Index of target detection
        """

        # Add target keypoints into a list
        matched_keypoints = [uav_keypoints[match.trainIdx] for match in matches]  # Append the corresponding UAV keypoint object

        # Clustering algorithm
        clusters = ut.fpnn(matched_keypoints, self.reference_point, nn_dist)

        clusters.sort(key=len)
        target_cluster = clusters[-1]

        uav_keyp_img = self.uav_image.copy()
        color = (0, 0, 0)
        for cluster in clusters:
            if cluster != target_cluster:
                while color == (0, 0, 255) or color == (0, 0, 0):
                    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                uav_keyp_img = cv.drawKeypoints(uav_keyp_img, cluster, outImage=None, color=(255, 255, 255))
            else:
                uav_keyp_img = cv.drawKeypoints(uav_keyp_img, cluster, outImage=None, color=(0, 0, 255))

        cv.imshow('UAV keypoints', uav_keyp_img)

        # Update the clustering reference point with the current largest cluster centroid
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
        """
        Gets mouse coordinates after clicking and appends said coordinates to bounding box variable
        """
        if event == cv.EVENT_LBUTTONDOWN and not self.clicked:
            self.bounding_box.append((x, y))
            self.clicked = True
        elif event == cv.EVENT_LBUTTONDOWN and self.clicked:
            print('Target captured. Press ESC')
            self.bounding_box.append((x, y))

    def set_up_target(self, cam_cap):
        """
        Args:
            cam_cap: VideoCapture object
        Returns: Image of captured object

        """
        while True:
            _, feed = cam_cap.read()
            cv.imshow('Target Selection', feed)
            # Wait for ESC
            if cv.waitKey(1) & 0XFF == 27:  # Save image of target scene locally
                cv.imwrite('app/datasets/targets/target.jpg', feed)
                cv.destroyAllWindows()
                break

        # Update the target scene image to contain only the selected region
        cv.namedWindow('Select ROI')
        cv.setMouseCallback('Select ROI', self._get_mouse_coord)
        target_frame = cv.imread('app/datasets/targets/target.jpg')
        while True:
            # Select target ROI
            cv.imshow('Select ROI', target_frame)
            if cv.waitKey(1) == 27:
                cv.destroyAllWindows()
                break

        # Store ROI coordinates
        x, y = self.bounding_box[0]
        w, h = self.bounding_box[1]
        w, h = w - x, h - y

        return target_frame[y: y + h, x: x + w]

# ------------------------------------------------------------------------------------------------------------------------------ #

# -------------------------------------------------- Video Feed Setup --------------------------------------------------------- #


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

# ------------------------------------------------------------------------------------------------------------------------------ #

# ------------------------------------------------ Initialization -------------------------------------------------------------- #

def initialize_homing(cam_URL=None, camera_index=-1, feature_extraction_method='ORB', n_features=10000, nn_dist=50):
    """
    Args:
        cam_URL: URL of IP or RTSP camera
        camera_index: Index of webcam for VideoCapture object
        feature_extraction_method: Selected feature extraction method. Can be 'SURF' or 'ORB'
        n_features: Number of features to be extracted from scene
        nn_dist: Minimum distance between keypoints for clustering algorithm
    """
    cap = None

    # Initialize camera feed
    if camera_index == -1 and cam_URL is None:
        for i in range(0, 10):
            cap = cv.VideoCapture(i)
            if cap.isOpened():
                camera_index = i
                break
    elif cam_URL is None:
        index_start = camera_index
        print(index_start)
        for i in range(index_start, 10):
            cap = cv.VideoCapture(i)
            if cap.isOpened():
                camera_index = i
                break
    else:
        camera_index = cam_URL
        cap = cv.VideoCapture(camera_index)

    # Initialize UI object
    ui = HomingUI()

    # Select target from video feed
    target_frame = ui.set_up_target(cap)
    cv.imshow('Cropped', target_frame)
    cv.waitKey(0)
    cap.release()

    # Initialize threaded camera
    threaded_cam = ThreadedCamera(camera_index)

    # Initialize Tracker object
    track = Tracker(target_frame, n_features, nn_dist)
    track.initialize_target(feature_extraction_method, target_frame)

    while True:
        start = time.time()

        # Perform object detection
        try:
            # Track target from previously captured frame
            res = track.track(threaded_cam.frame, feature_extraction_method)
            if res:
                res_frame = ut.draw_image(threaded_cam.frame, res[0], res[1], 5, (0, 0, 255))
            else:
                res_frame = threaded_cam.frame
            # Display result
            threaded_cam.show_frame(res_frame)
            # print('Operations took {:2f}s'.format(time.time() - start))

        except AttributeError:
            pass

# ------------------------------------------------------------------------------------------------------------------------------ #
