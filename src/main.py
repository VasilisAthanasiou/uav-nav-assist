import cv2 as cv
import numpy as np
import os
import imutils
from matplotlib import pyplot as plt


# -------------------------------------------------- Misc Methods -------------------------------------------------------------- #

def wait_for_esc():
    while True:
        if cv.waitKey(1) == 27:
            cv.destroyAllWindows()
            break


def yes_no(arg):
    args = (['y', 'Y', 'yes', 'Yes', 'YES'], ['n', 'N', 'no', 'No', 'NO'])
    while True:
        if arg in args[0]:
            return True
        elif arg in args[1]:
            return False
        else:
            print("Please give a correct answer: ['y', 'Y', 'yes', 'Yes', 'YES'] or ['n', 'N', 'no', 'No', 'NO']")


# ------------------------------------------------------------------------------------------------------------------------------ #

# ------------------------------------------------- Image Processing ----------------------------------------------------------- #

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


# ------------------------------------------------------------------------------------------------------------------------------ #

# --------------------------------------------- Matching Algorithm ------------------------------------------------------------- #

# Returns the top left pixel location of the sensed template
def findTarget(src, temp):
    """Uses openCV's matchTemplate to find a desired target inside the source image.

    :param src: Source image
    :param temp: Template image
    :return: Tuple containing the sensed target top left coordinates
    """

    # Apply template matching
    res = cv.matchTemplate(src, temp, cv.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    return max_loc  # Return top left position


# ------------------------------------------------------------------------------------------------------------------------------ #

# --------------------------------------------- Statistical Analysis ----------------------------------------------------------- #

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

        for img in self.src:

            processed_img = prc.process_image(img, args=['grayscale'])
            error_temp_img = 0  # Error for one template. How displaced the sensed frame is from the actual one

            for i in range(int(n_templates / len(self.src))):
                # Store both actual and sensed x and y
                actual_x, actual_y, _ = self.actual_match[i].split(',')
                sensed_x, sensed_y = findTarget(processed_img, prc.process_image(self.temp[counter], rot_deg=self.rotation,
                                                                                 args=['grayscale', 'rotate']))

                # Error for one template is the added absolute differences between x and y divided the number of pixels
                error_temp_img += np.abs(int(sensed_x) - int(actual_x)) + np.abs(int(sensed_y) - int(actual_y))
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
        file = open("../experiment-results.txt", "a")
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


# ------------------------------------------------------------------------------------------------------------------------------ #

# -------------------------------------------------- Simulation ---------------------------------------------------------------- #
class Simulator:

    def __init__(self, sat_images=None, sim_uav_images=None):
        self.sat_images = sat_images
        self.sim_uav_images = sim_uav_images
        self.params = 100, 'East', 'South', 45, 200
        self.center_displacement = (0, 0)

    def _set_uav_params(self, use_defaults):
        if not use_defaults:
            dist_center = input('Enter distance from center\n')
            dx_bias = input('Enter direction of horizontal displacement: West or East\n')
            dy_bias = input('Enter direction of vertical displacement: North or South\n')
            heading = int(input('Enter heading angle relative to true North\n'))
            capture_dim = int(input('Enter the dimension of the captured images. Default is 200\n'))
            self.params = int(dist_center), dx_bias, dy_bias, int(heading), int(capture_dim)

    def simulate(self, sat_dir, sim_uav_dir, use_defaults):
        # Initialize variables
        self._set_uav_params(use_defaults)
        dist_center, dx_bias, dy_bias, heading, capture_dim = self.params
        prc = Processor()

        # Read images
        rd = ImageReader()
        sat_images = rd.readImages(sat_dir)
        sim_uav_images = rd.readImages(sim_uav_dir)

        dx = {'West': -dist_center, 'East': dist_center}
        dy = {'South': dist_center, 'North': -dist_center}
        dx = dx[dx_bias]
        dy = dy[dy_bias]

        self.sat_images = sat_images
        self.sim_uav_images = sim_uav_images

        # Simulation loop
        for index in range(len(self.sat_images)):
            # Set center of satellite image
            sat_image_center = (int(self.sat_images[index].shape[0] / 2), int(self.sat_images[index].shape[1] / 2))

            # Simulate heading errors
            inertial_error = np.random.uniform(0, 2)

            # Rotate the image clockwise
            uav_processed_image = prc.process_image(sim_uav_images[index], rot_deg=heading + inertial_error, args=['rotate'])
            uav_prc_center = (int(uav_processed_image.shape[0] / 2), int(uav_processed_image.shape[1] / 2))
            print('The INS made a {} degree error\nPress ESC to continue'.format(inertial_error))

            # Define center of captured image inside the rotated uav image
            capt_img_rotated_center = (uav_prc_center[0] + dx, uav_prc_center[1] + dy)

            # Calculate the center of the captured image relative to the satellite source (uav_processed_image before rotation)
            x = capt_img_rotated_center[0]
            y = capt_img_rotated_center[1]
            p = uav_prc_center[0]
            q = uav_prc_center[1]
            theta = np.deg2rad(heading)

            # Finding coordinates of capture center
            actual_capture_coord = (x - p) * np.cos(-theta) + (y - q) * np.sin(-theta) + p, -(x - p) * np.sin(-theta) + (
                    y - q) * np.cos(-theta) + q

            # "Capturing" the UAV image by cropping the uav_processed_image
            capt_top_left = (capt_img_rotated_center[0] - int(capture_dim / 2),  # Top left pixel location of captured image
                             capt_img_rotated_center[1] - int(capture_dim / 2))

            captured_img = uav_processed_image[capt_top_left[1]:capt_top_left[1] + capture_dim,
                           capt_top_left[0]:capt_top_left[0] + capture_dim]

            cv.imshow('Captured image', captured_img)
            wait_for_esc()

            print('INS : {} degrees insertion angle\nRotating image accordingly...\nPress ESC to continue'.format(heading))
            captured_img = imutils.rotate(captured_img, -heading)

            # Crop the image to get rid of black areas caused by rotation
            captured_img = captured_img[
                           int(captured_img.shape[0] / 4):int(captured_img.shape[0] / 4) + int(captured_img.shape[0] / 2),
                           int(captured_img.shape[1] / 4):int(captured_img.shape[1] / 4) + int(captured_img.shape[1] / 2)]
            cv.imshow('INS corrected captured image', captured_img)
            wait_for_esc()

            # Find where the captured image is located relative to the satellite image
            captured_image_location = findTarget(self.sat_images[index], captured_img)  # Top-left location of the template image

            captured_img_center = (captured_image_location[0] + int(captured_img.shape[0] / 2),
                                   captured_image_location[1] + int(captured_img.shape[1] / 2))

            # Send the course correction signal
            self.center_displacement = captured_img_center[0] - sat_image_center[0], sat_image_center[1] - captured_img_center[1]

            print('The UAV is off center {} meters horizontally and {} meters vertically\n'
                  'And the error is {:.2f} meters horizontally and {:.2f} meters vertically\n\n'.format(
                self.center_displacement[0], self.center_displacement[1],
                np.abs(captured_img_center[0] - actual_capture_coord[0]),
                np.abs(captured_img_center[1] - actual_capture_coord[1])))


# ------------------------------------------------------------------------------------------------------------------------------ #

# -------------------------------------------------- Setup Data ---------------------------------------------------------------- #

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


# ------------------------------------------------------------------------------------------------------------------------------ #

# -------------------------------------------------- User Interface ------------------------------------------------------------ #

class UI:

    def __init__(self, method=''):
        self._method = method
        self.simulator = None
        self.evaluator = None
        self.cwd = '../datasets'

    def experiment(self, method):
        """

        Args:
            method: Selects the method to run. Can be = 'simulation', 'plot', 'write text'
        """
        self._method = method
        return self._get_method()


    def _get_method(self):

        if self._method == 'simulation':
            return self._use_simulation()
        elif (self._method == 'plot') or (self._method == 'write text'):
            return self._use_evaluation()


    def _use_simulation(self):

        self.cwd += '/sources/' + input(' the desired dataset directory\n{}\n'.format(os.listdir(self.cwd + '/sources/')))
        sat_dir = self.cwd + '/' + input('Select satellite image source:\n{}\n'.format(os.listdir(self.cwd))) + '/'
        uav_dir = self.cwd + '/' + input('Select UAV image source:\n{}\n\n'.format(os.listdir(self.cwd)))
        use_defaults = yes_no(input('Do you want to use the default simulation values? : '))

        self.simulator = Simulator()

        return self.simulator.simulate(sat_dir, uav_dir, use_defaults)


    def _use_evaluation(self):
        src_dir = self.cwd + '/sources/'
        tmp_dir = self.cwd + '/templates/'
        src_dir += input('Select a source directory\n{}\n'.format(os.listdir(src_dir)))
        src_dir += '/' + input('Specify the source directory\n{}\n'.format(os.listdir(src_dir)))
        tmp_dir += input('Select a template directory\n{}\n'.format(os.listdir(tmp_dir)))
        act_txt_path = tmp_dir + '/' + [file if '.txt' in file else None for file in os.listdir(tmp_dir)][0]
        tmp_dir += '/' + input('Specify the template directory\n{}\n'.format(os.listdir(tmp_dir)))
        rot = int(input('Enter the template rotation\n'))

        self.evaluator = Evaluator(src_dir, tmp_dir, act_txt_path, rotation=rot)

        return self.evaluator.evaluate(self._method)


# ------------------------------------------------------------------------------------------------------------------------------ #

# ----------------------------------------------------- Main ------------------------------------------------------------------- #

ui = UI()
ui.experiment('simulation')  # Either 'simulation', 'plot' or 'write text'

# ------------------------------------------------------------------------------------------------------------------------------ #
