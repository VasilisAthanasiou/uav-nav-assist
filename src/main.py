import cv2 as cv
import numpy as np
import os
import imutils
from matplotlib import pyplot as plt


# --------------------------------------------- Image Processing --------------------------------------------------------- #

def process_image(src_img, rot_deg=0):
    """Applies image processing techniques on an image.

    :param src_img: Image to be processed
    :param rot_deg: Degrees that the image will be rotated
    :return: Returns the processed image

    """

    # Convert to grayscale
    try:
        processed_img = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)
    except cv.error:
        # The image was probably already converted to grayscale
        processed_img = src_img

    # Rotate the image
    if rot_deg != 0:
        processed_img = imutils.rotate(processed_img, int(rot_deg))

    return processed_img


# ------------------------------------------------------------------------------------------------------------------------ #

# ------------------------------------------ Matching Algorithm ---------------------------------------------------------- #

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


# ------------------------------------------------------------------------------------------------------------------------ #

# ------------------------------------------ Statistical Analysis -------------------------------------------------------- #

class Evaluator:
    """Performs the find_target() function for multiple source images on multiple templates, compares the results with
    the locations in the actual_match list, calculates the displacement and finally; writes the results on a text file.

    :param src: Source images list
    :param temp: Template images list
    :param actual_match: List of strings containing the correct coordinates for a give target
    :param rotation: Rotation value for process_image()
    :return: A well structured string containing the results of the experiment or a tuple of the resulting data
    """
    def __init__(self, src, temp, actual_match, rotation=0):
        self.src = readImages(src)
        self.temp = readImages(temp)
        self.actual_match = open(actual_match, 'r').readlines()
        self.rotation = rotation
        self.num_of_images = len(src)
        self.n_templates = len(temp)
        self.error = ''
        self.result = ''

    def evaluate(self, method):
        if method == 'write-txt':
            return self._write_experiment()
        elif method == 'plot':
            return self._plot_data()

    def _run_evaluation(self):

        self.error_img = []  # Error for each image Sum(error for every template / number of templates)
        counter = 0  # Keeps track of each iterations
        result = ''

        for img in self.src:

            processed_img = process_image(img)
            error_temp_img = 0  # Error for one template. How displaced the sensed frame is from the actual one

            for i in range(int(self.n_templates / self.num_of_images)):
                # Store both actual and sensed x and y
                actual_x, actual_y, _ = self.actual_match[i].split(',')
                sensed_x, sensed_y = findTarget(processed_img, process_image(self.temp[counter], rot_deg=self.rotation))

                # Error for one template is the added absolute differences between x and y divided the number of pixels
                error_temp_img += np.abs(int(sensed_x) - int(actual_x)) + np.abs(int(sensed_y) - int(actual_y))
                counter += 1

            self.error_img.append(error_temp_img / self.n_templates)  # Error for a whole image tested with multiple templates
            self.result += ("Mean error for image {} : {}px\n".format(int(counter / self.n_templates),
                                                                 error_temp_img / self.n_templates))
        # Error for all images
        self.error = sum(self.error_img) / self.num_of_images
        self.result += ("There is {} pixel mean error.\n\n".format(self.error))


    def _write_experiment(self):
        # Write the experiment results on a text file
        self._run_evaluation()
        file = open("../experiment-results.txt", "a")
        file.write("-------------- Results using {}deg rotation on source images on dataset --------------\n{}".format(self.rotation, self.result))


    def _plot_data(self):
        self._run_evaluation()
        results = ['{}'.format(counter + 1) for counter in range(len(self.error_img))], [round(self.error_img[counter], 2) for counter
                                                                                 in range(len(self.error_img))], self.error
        counter = 0
        colors = ['b', 'g', 'r', 'c', 'm', 'y', '#3277a8', '#a87332', '#915e49']
        for result in results:
            fig = plt.figure(counter)
            plt.bar(result[0], result[1], color=colors[counter])
            for index, value in enumerate(result[1]):
                plt.text(index, value, str(value))

            plt.xlabel('Images')
            plt.ylabel('Mean pixel error')
            plt.axis([0, self.num_of_images, 0, max(self.error_img)+ 100])
            ax = plt.gca()
            ax.set_axisbelow(True)
            plt.gca().yaxis.grid(linestyle="dashed")
            plt.show()
            counter += 1


# ------------------------------------------------------------------------------------------------------------------------ #

# ----------------------------------------------- Simulation ------------------------------------------------------------- #
def simulate(sat_images, sim_uav_images, d_error=100, dx_bias='East', dy_bias='South', heading=45, capture_dim=200):
    """Runs a flight simulation.

    Args:
        sat_images: Set of satellite images, that represent images stored on the UAV
        sim_uav_images : Set of images that
        d_error: Displacement error. The error caused by the INS inaccuracy
        dx_bias: Direction of the horizontal error. Is always fixed for a give type of inertial system
        dy_bias: Direction of the vertical error. Is always fixed for a given type of inertial system
        heading: Angle from which the UAV will be inserted into the satellite image.
        capture_dim: Dimension of image captured by the UAV
    Returns:

    """

    # Initialize variables
    dx = {'West': -d_error, 'East': d_error}
    dy = {'South': d_error, 'North': -d_error}
    dx = dx[dx_bias]
    dy = dy[dy_bias]

    # Simulation loop
    for index in range(len(sat_images)):

        # Set center of satellite image
        sat_image_center = (int(sat_images[index].shape[0] / 2), int(sat_images[index].shape[1] / 2))

        # Simulate heading errors
        inertial_error = np.random.uniform(0, 2)

        # Rotate the image clockwise
        uav_processed_image = imutils.rotate(sim_uav_images[index], heading + inertial_error)
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

        # Finding coordinates of capture center by
        actual_capture_coord = (x - p) * np.cos(-theta) + (y - q) * np.sin(-theta) + p, -(x - p) * np.sin(-theta) + (
                y - q) * np.cos(-theta) + q

        # "Capturing" the UAV image by cropping the uav_processed_image
        capt_top_left = (capt_img_rotated_center[0] - int(capture_dim / 2),  # Top left pixel location of captured image
                         capt_img_rotated_center[1] - int(capture_dim / 2))

        captured_img = uav_processed_image[capt_top_left[1]:capt_top_left[1] + capture_dim,
                       capt_top_left[0]:capt_top_left[0] + capture_dim]

        cv.imshow('Captured image', captured_img)
        wait_for_ESC()

        print('INS : {} degrees insertion angle\nRotating image accordingly...\nPress ESC to continue'.format(heading))
        captured_img = imutils.rotate(captured_img, -heading)

        # Crop the image to get rid of black areas caused by rotation
        captured_img = captured_img[
                       int(captured_img.shape[0] / 4):int(captured_img.shape[0] / 4) + int(captured_img.shape[0] / 2),
                       int(captured_img.shape[1] / 4):int(captured_img.shape[1] / 4) + int(captured_img.shape[1] / 2)]
        cv.imshow('INS corrected captured image', captured_img)
        wait_for_ESC()

        # Find where the captured image is located relative to the satellite image
        captured_image_location = findTarget(sat_images[index], captured_img)  # Top-left location of the template image

        captured_img_center = (captured_image_location[0] + int(captured_img.shape[0] / 2),
                               captured_image_location[1] + int(captured_img.shape[1] / 2))

        # Send the course correction signal
        course_displacement = captured_img_center[0] - sat_image_center[0], sat_image_center[1] - captured_img_center[1]

        print('The UAV is off center {} meters horizontally and {} meters vertically\n'
              'And the error is {:.2f} meters horizontally and {:.2f} meters vertically\n\n'.format(
            course_displacement[0], course_displacement[1],
            np.abs(captured_img_center[0] - actual_capture_coord[0]),
            np.abs(captured_img_center[1] - actual_capture_coord[1])))


# ------------------------------------------------------------------------------------------------------------------------ #

# ----------------------------------------------- Setup Data ------------------------------------------------------------- #

# Reads images and converts them to grayscale
def readImages(directory):
    # Append all the paths into lists
    img_paths = [os.path.join(directory, image_path) for image_path in os.listdir(directory)]
    img_paths.sort()

    # Print all paths to make sure everything is ok
    for path in img_paths:
        print(path)

    return [cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2GRAY) for img_path in img_paths]


# ------------------------------------------------------------------------------------------------------------------------ #

# ----------------------------------------------- Misc Methods ----------------------------------------------------------- #

def wait_for_ESC():
    while True:
        if cv.waitKey(1) == 27:
            cv.destroyAllWindows()
            break

class UI:

    def experiment(self, method):
        return self._get_method(method)

    def _get_method(self, method):
        cwd = '../datasets'
        if method == 'simulation':
            cwd += '/sources/' + input('Please type in the desired dataset directory\n{}\n'.format(os.listdir(cwd)))
            sat_dir = cwd + '/' + input('Select satellite image source:\n{}\n'.format(os.listdir(cwd))) + '/'
            uav_dir = cwd + '/' + input('Select UAV image source:\n{}\n\n'.format(os.listdir(cwd)))
            head = int(input("Type in the UAV's heading : "))
            dist = int(input("Type in the distance from the center. This will be applied on both axes :\n"))

            return simulate(readImages(sat_dir), readImages(uav_dir), d_error=dist, heading=head)

        else:
            src_dir = cwd + '/sources/'
            tmp_dir = cwd + '/templates/'
            src_dir += input('Please select a source directory\n{}\n'.format(os.listdir(src_dir)))
            src_dir += '/' + input('Please specify the source directory\n{}\n'.format(os.listdir(src_dir)))
            tmp_dir += input('Please select a template directory\n{}\n'.format(os.listdir(tmp_dir)))
            act_txt_path = tmp_dir + '/' + [file if '.txt' in file else None for file in os.listdir(tmp_dir)][0]
            tmp_dir += '/' + input('Please specify the template directory\n{}\n'.format(os.listdir(tmp_dir)))
            rot = input('Enter the template rotation\n')

            evaluator = Evaluator(src_dir, tmp_dir, act_txt_path, rotation=rot)

            return evaluator.evaluate(method)


# ------------------------------------------------------------------------------------------------------------------------ #

# -------------------------------------------------- Main ---------------------------------------------------------------- #

ui = UI()
ui.experiment('write-txt')

# ---------------------------------------------------------------------------------------------------------------------- #




