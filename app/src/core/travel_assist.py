import cv2 as cv
import time
import numpy as np
import os
import imutils
from matplotlib import pyplot as plt
import app.src.unautil.utils as ut


# ------------------------------------------------------- Image Processing --------------------------------------------------------------- #

def _grayscale(img):
    # Convert to grayscale
    try:
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    except cv.error:
        # The image was probably already converted to grayscale
        return img


def _rotate(img, rot_deg):
    # Rotate the image
    return imutils.rotate(img, int(rot_deg))


class Processor:

    def __init__(self):
        self.processed_img = None

    def process_image(self, src_img, rot_deg=0, args=None):
        """Applies image processing techniques on an image.
        Args:
            src_img: Image to be processed
            rot_deg: Degrees that the image will be rotated
            args: List of strings containing the functions that can be called. Can be 'rotate' and 'grayscale'
        Returns:
            Processed image

        """
        self.processed_img = src_img
        if 'rotate' in args:
            self.processed_img = _rotate(self.processed_img, rot_deg)
        elif 'grayscale' in args:
            self.processed_img = _grayscale(self.processed_img)

        return self.processed_img


# ---------------------------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------- Matching Algorithm ------------------------------------------------------------------ #

class Matcher:

    def __init__(self):
        self.src = None
        self.temp = None

    # Returns the top left pixel location of the sensed template
    def findTarget(self, src, temp, method):
        """
        Uses openCV's matchTemplate to find a desired target inside the source image.
        Args:
            src: Source image
            temp: Template image
            method: Method to use for matching. Options are : 'template-matching

        Returns:

        """
        self.src = src
        self.temp = temp

        return self._select_matcher(method)

    def _select_matcher(self, arg):
        if arg == 'template matching':
            return self._template_matching()
        elif arg == 'sift matching':
            return self._sift_matching()
        elif arg == 'fast matching':
            return self._fast_feature_detector()

    def _template_matching(self):
        # Apply template matching
        start_time = time.time()
        prev_max = 0
        # This coefficient prevents the algorithm from getting stuck in a loop
        cycle_prevent_coeff = 0.016

        res = cv.matchTemplate(self.src, self.temp, cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        # Zoom the image out initially
        image = self.temp
        resize_value = 0.9
        zoom_out = True
        max_vals = []

        # Perform template matching on different sizes of the template image to find the highest correlation value
        while max_val < 0.42:
            image = imutils.resize(self.temp, int(self.temp.shape[0] * resize_value), int(image.shape[1] * resize_value))
            res = cv.matchTemplate(self.src, image, cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            print('Image shape : {} , Max value : {} '.format(image.shape, max_val))
            max_vals.append((max_val, image.shape))

            if zoom_out:
                resize_value -= 0.1 # Zoom out value
            else:
                resize_value += 0.1 # Zoom in value

            if prev_max > max_val + cycle_prevent_coeff:  # If the previous correlation is higher that the current, then change zoom
                # Zoom the image in
                resize_value = 1.1
                image = self.temp
                zoom_out = False
            prev_max = max_val
            if time.time() - start_time > 1.0:  # If this processes lasts for more that 1 sec then end the process
                _, (width, height) = max(max_vals, key=lambda value: value[0])
                image = imutils.resize(self.temp, width, height)
                res = cv.matchTemplate(self.src, image, cv.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
                break

        end_time = time.time()
        print('Template Matching took {:.2f}s\nImage shape {}'.format((end_time - start_time), image.shape))
        return max_loc[0], max_loc[1], max_val  # Return top left position

    def _sift_matching(self):
        """Performs the Scale Invariant Feature Transform on two images and matches each corresponding keypoint

        Returns:

        """
        start_time = time.time()
        sift = cv.xfeatures2d.SIFT_create()

        keyp_src, desc_src = sift.detectAndCompute(self.src, None)
        keyp_temp, desc_temp = sift.detectAndCompute(self.temp, None)

        # Feature matching
        bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)

        end_time = time.time()
        print('Sift Matching took {:.2f}s'.format((end_time - start_time)))

        matches = sorted(bf.match(desc_src, desc_temp), key=lambda x: x.distance)

        res = cv.drawMatches(self.src, keyp_src, self.temp, keyp_temp, matches[:50], self.temp, flags=2)
        plt.imshow(res), plt.show()

    def _fast_feature_detector(self):

        """Performs the FAST (Features from Accelerated Segment Test) feature extraction method on an image and displays
        the detected keypoints

        Returns:

        """
        start_time = time.time()

        fast = cv.FastFeatureDetector_create()

        keyp_src = fast.detect(self.src, None)
        keyp_temp = fast.detect(self.temp, None)

        res = cv.drawKeypoints(self.src, keyp_src, self.src, color=(255, 0, 0)), cv.drawKeypoints(self.temp, keyp_temp, self.temp,
                                                                                                  color=(0, 0, 255))

        end_time = time.time()
        print('Fast Matching took {:.2f}s'.format((end_time - start_time)))

        fig = plt.figure()
        ax1 = fig.add_subplot(2, 2, 1)
        plt.imshow(res[0])
        ax2 = fig.add_subplot(2, 2, 2)
        plt.imshow(res[1])
        plt.show()


# ---------------------------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------- Statistical Evaluation --------------------------------------------------------- #

class Evaluator:
    """Performs the find_target() function for multiple source images on multiple templates, compares the results with
    the locations in the actual_match list, calculates the displacement and finally; writes the results on a text file.
    Args:
        src: Source images list
        temp: Template images list
        actual_match: List of strings containing the correct coordinates for a give target
        rotation: Rotation value for process_image()
    Returns: A well structured string containing the results of the experiment or a tuple of the resulting data
    """

    def __init__(self, src, temp, actual_match, rotation=0):
        rd = ImageReader()
        self.src = rd.readImages(src)
        self.temp = rd.readImages(temp)
        self.actual_match = open(actual_match, 'r').readlines()
        self.rotation = rotation
        self.error = 0.0
        self.result_txt = ''

    def evaluate(self, method):
        if method == 'write text':
            return self._write_experiment()
        elif method == 'plot':
            return self._plot_data()

    def _run_evaluation(self):
        self.img_error = []  # Error for each image Sum(error for every template / number of templates)
        n_templates = len(self.temp)
        print(n_templates)
        counter = 0  # Keeps track of each iterations
        prc = Processor()
        matcher = Matcher()

        for img in self.src:

            processed_img = prc.process_image(img, args=['grayscale'])
            error_temp_img = 0  # Error for one template. How displaced the sensed frame is from the actual one

            for i in range(int(n_templates / len(self.src))):
                # Store both actual and sensed x and y
                actual_x, actual_y, _ = self.actual_match[i].split(',')
                sensed_x, sensed_y, value = matcher.findTarget(processed_img, prc.process_image(self.temp[counter], rot_deg=self.rotation,
                                                                                 args=['grayscale', 'rotate']), method='template matching')

                # Error for one template is the added absolute differences between x and y divided the number of pixels
                error_temp_img += np.abs(int(sensed_x) - int(actual_x)) + np.abs(int(sensed_y) - int(actual_y))

                self.result_txt += 'Error for template {} : {}\nTemplate max value : {:.2f}\n'.format(self.temp[counter].shape, np.abs(
                    int(sensed_x) - int(actual_x)) + np.abs(int(sensed_y) - int(actual_y)), value) + '\n'
                counter += 1


            self.img_error.append(error_temp_img / n_templates)  # Error for a whole image tested with multiple templates
            self.result_txt += ("Mean error for image {} : {}px\n".format(int(counter / n_templates),
                                                                          error_temp_img / n_templates))
        # Error for all images
        self.error = sum(self.img_error) / len(self.src)
        self.result_txt += ("There is {} pixel mean error.\n\n".format(self.error))

    def _write_experiment(self):
        # Write the experiment results on a text file
        self._run_evaluation()
        file = open("../datasets/experiment-results.txt", "a")
        file.write(
            "-------------- Results using {}deg rotation on source images on dataset --------------\n{}".format(self.rotation,
                                                                                                                self.result_txt))

    def _plot_data(self):
        self._run_evaluation()
        results = [round(self.img_error[counter], 2) for counter in range(len(self.img_error))]

        colors = ['b', 'g', 'r', 'c', 'm', 'y', '#3277a8', '#a87332', '#915e49']

        plt.xlabel('Images')
        plt.ylabel('Mean pixel error')
        plt.axis([0, len(self.src), 0, 700])
        ax = plt.gca()
        ax.set_axisbelow(True)
        plt.gca().yaxis.grid(linestyle="dashed")
        for index, value in enumerate(results):
            plt.text(index, value, str(value))  # Problem with this stupid thing
        plt.bar(results[0], results[1], color=colors[np.random.randint(0, 8)])
        plt.show()


# ---------------------------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------ Simulation ---------------------------------------------------------------------- #

class Simulator:

    def __init__(self, sat_images=None, sim_uav_images=None):
        self.sat_images = sat_images
        self.sim_uav_images = sim_uav_images
        self.params = 100, 'East', 'North', 45, 200
        self.dx = 0
        self.dy = 0

    def _verbose_sim(self, sat_image, sim_uav_image, inertial_error):
        """ Performs simulation with detailed output, including images and console prints

        Args:
            sat_image: Satellite image
            sim_uav_image: UAV image
            inertial_error: Heading error produced by the INS

        Returns:

        """
        _, dx_bias, dy_bias, heading, capture_dim = self.params

        prc = Processor()
        matcher = Matcher()
        # Set center of satellite images
        sat_image_center = (int(sat_image.shape[0] / 2), int(sat_image.shape[1] / 2))

        # Coordinates where the UAV will capture an image
        actual_capture_coord = sat_image_center[0] + self.dx, sat_image_center[1] + self.dy
        print('The UAV is off course {} horizontally and {} vertically'.format(self.dx, self.dy))

        # "Capturing" the UAV image by cropping the uav_processed_image
        capt_top_left = (actual_capture_coord[0] - int(capture_dim / 2),  # Top left pixel location of captured image
                         actual_capture_coord[1] - int(capture_dim / 2))
        # Cropping the UAV image
        captured_img = ut.snap_image(sim_uav_image, capt_top_left[0], capt_top_left[1], dim=capture_dim)

        # Displaying the actual UAV location on the satellite image
        marked_loc_img = ut.draw_image(sat_image, actual_capture_coord[0], actual_capture_coord[1], color=(255, 0, 0))
        cv.imshow('Actual UAV location', imutils.resize(marked_loc_img, 500, 500))
        ut.wait_for_esc()

        # Rotate the image clockwise to simulate the insertion angle plus the INS error
        captured_img = prc.process_image(captured_img, rot_deg=heading + inertial_error, args=['rotate'])
        print('The INS made a {} degree error\nPress ESC to continue'.format(inertial_error))

        # Rotate the image to match its orientation into what the UAV thinks is true north. Doesn't include the INS error
        print('INS : {} degrees insertion angle\nRotating image accordingly...\nPress ESC to continue'.format(heading))
        captured_img = prc.process_image(captured_img, rot_deg=-heading, args=['rotate'])

        # Crop the image to get rid of black areas caused by rotation
        captured_img = captured_img[
                       int(captured_img.shape[0] / 4):int(captured_img.shape[0] / 4) + int(captured_img.shape[0] / 2),
                       int(captured_img.shape[1] / 4):int(captured_img.shape[1] / 4) + int(captured_img.shape[1] / 2)]

        # Find where the captured image is located relative to the satellite image
        cap_location_x, cap_location_y, value = matcher.findTarget(sat_image, captured_img,
                                                                   method='template matching')  # Top-left location of the template image

        # captured_image_location contains the top left pixel location of matched image. Calculate the central pixel
        captured_img_center = (cap_location_x + int(captured_img.shape[0] / 2),
                               cap_location_y + int(captured_img.shape[1] / 2))

        # Diplay both the actual and the calculated UAV position on the image
        marked_loc_img = ut.draw_image(marked_loc_img, captured_img_center[0], captured_img_center[1])
        cv.imshow('Actual location (Blue) vs Calculated location (Red)', imutils.resize(marked_loc_img, 500, 500))
        ut.wait_for_esc()

        # Course correction data
        x_error, y_error = captured_img_center[0] - actual_capture_coord[0], captured_img_center[1] - actual_capture_coord[1]
        sensed_center_displacement = captured_img_center[0] - sat_image_center[0], sat_image_center[1] - captured_img_center[1]
        print('The UAV is off center {} meters horizontally and {} meters vertically\n'
              'And the error is {:.2f} meters horizontally and {:.2f} meters vertically\n\n'.format(
            sensed_center_displacement[0], sensed_center_displacement[1], x_error, y_error))

        return x_error, y_error

    def _set_uav_params(self, use_defaults):
        if not use_defaults:
            dist_center = input('Enter distance from center\n')
            dx_bias = input('Enter direction of horizontal displacement: West or East\n')
            dy_bias = input('Enter direction of vertical displacement: North or South\n')
            heading = int(input('Enter heading angle relative to true North\n'))
            capture_dim = int(input('Enter the dimension of the captured images. Default is 200\n'))
            self.params = int(dist_center), dx_bias, dy_bias, int(heading), int(capture_dim)

    def _init_variables(self, sat_dir, sim_uav_dir, use_defaults):
        """

        Args:
            sat_dir: Satellite images directory
            sim_uav_dir: UAV images directory
            use_defaults: Determines whether to use default Simulation.params values or have the user set them

        Returns:

        """
        # Initialize variables
        self._set_uav_params(use_defaults)
        dist_center, dx_bias, dy_bias, heading, capture_dim = self.params

        # Read images
        rd = ImageReader()
        self.sat_images = rd.readImages(sat_dir)
        self.sim_uav_images = rd.readImages(sim_uav_dir)

        # Central displacement error
        dx = {'West': -dist_center, 'East': dist_center}
        dy = {'South': dist_center, 'North': -dist_center}
        self.dx = dx[dx_bias]
        self.dy = dy[dy_bias]

        return dist_center, dx_bias, dy_bias, heading, capture_dim

    def simulate(self, sat_dir, sim_uav_dir, use_defaults, inertial_error=np.random.uniform(0, 2)):
        """

        Args:
            sat_dir: Directory of satellite images
            sim_uav_dir: Directory of UAV images
            use_defaults: Determines whether to use default Simulation.params values or have the user set them
            inertial_error: Simulates the heading inaccuracy caused by the INS

        Returns:

        """
        self._init_variables(sat_dir, sim_uav_dir, use_defaults)

        # Simulation loop
        for index in range(len(self.sat_images)):
            # Run a simulation with _verbose_sim. Another method could also be used
            try:
                x_error, y_error = self._verbose_sim(self.sat_images[index], self.sim_uav_images[index], inertial_error)
            except cv.error as e:
                print('UAV flew outside the expected region\n\n{}'.format(e))
                return
            # Accumulate the central displacement error
            self.dx += x_error
            self.dy += y_error


# ---------------------------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------ Setup Data ---------------------------------------------------------------- #

# Reads images and converts them to grayscale
class ImageReader:

    def __init__(self):
        self.directory = ''

    def readImages(self, directory):
        """Reads all images from selected path using OpenCV's imread and converts them to grayscale.

        Args:
            directory: Images directory

        Returns:

        """
        self.directory = directory

        # Append all the paths into lists
        img_paths = [os.path.join(directory, image_path) for image_path in os.listdir(directory)]

        # Sort the paths
        try:
            img_paths.sort(key=lambda name: int(os.path.splitext(os.path.basename(name))[0]))
        except ValueError:
            img_paths.sort()

        # Print all paths to make sure everything is ok
        for path in img_paths:
            print(path)

        return [cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2GRAY) for img_path in img_paths]


# ---------------------------------------------------------------------------------------------------------------------------------------- #

# --------------------------------------------------------- User Interface --------------------------------------------------------------- #

class TravelUI(ut.UI):

    def __init__(self, method=''):
        super(TravelUI, self).__init__()
        self.simulator = None
        self.evaluator = None

    def _use_simulation(self):
        self.cwd += '/sources/' + input(' the desired dataset directory\n{}\n'.format(os.listdir(self.cwd + '/sources/')))
        sat_dir = self.cwd + '/' + input('Select satellite image source:\n{}\n'.format(os.listdir(self.cwd))) + '/'
        uav_dir = self.cwd + '/' + input('Select UAV image source:\n{}\n\n'.format(os.listdir(self.cwd)))
        use_defaults = ut.yes_no(input('Do you want to use the default simulation values? : '))

        self.simulator = Simulator()

        return self.simulator.simulate(sat_dir, uav_dir, use_defaults)

    def _use_evaluation(self):
        src_dir = self.cwd + '/sources/'
        tmp_dir = self.cwd + '/templates/'
        src_dir += input('Select a source directory\n{}\n'.format(os.listdir(src_dir)))
        src_dir += '/' + input('Specify the source directory\n{}\n'.format(os.listdir(src_dir)))
        tmp_dir += input('Select a template directory\n{}\n'.format(os.listdir(tmp_dir)))
        act_txt_path = tmp_dir + '/' + [file if '.txt' in file else None for file in os.listdir(tmp_dir)][0]
        tmp_dir += '/images/'
        rot = int(input('Enter the template rotation\n'))

        self.evaluator = Evaluator(src_dir, tmp_dir, act_txt_path, rotation=rot)

        return self.evaluator.evaluate(self._method)


# ---------------------------------------------------------------------------------------------------------------------------------------- #


# ----------------------------------------------------------- Main ----------------------------------------------------------------------- #
# ui = TravelUI()
# ui.experiment('simulation')  # Either 'simulation', 'plot' or 'write text'

sim = Simulator()
sim.simulate('../datasets/travel-assist/sources/source-diverse/3.cloudy-images',
             '../datasets/travel-assist/sources/source-diverse/2.blurred', True)

# ---------------------------------------------------------------------------------------------------------------------------------------- #
