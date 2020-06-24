import cv2 as cv
import numpy as np
import os
#from scipy import ndimage
import imutils

from matplotlib import pyplot as plt


# ---------------------------------------------Image Processing--------------------------------------------------------- #

def process_image(src_img, resize=-1, rot_deg=0):
    '''Applies image processing techniques on an image.

    :param src_img: Image to be processed
    :param resize: Resize value. Takes float values between 0 and 1.
    :param rot_deg: Degrees that the image will be rotated
    :return: Returns the processed image

    '''

    # Resize image
    if resize != -1:
        res_img = cv.resize(src_img, None, fx=resize, fy=resize, interpolation=cv.INTER_CUBIC)

    # Convert to grayscale
    gray_image = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)

    processed_img = gray_image

    # Rotate the image
    if rot_deg != 0:
        processed_img = ndimage.rotate(gray_image, rot_deg)

    return processed_img


# ---------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------ Matching Algorithm -------------------------------------------------------- #

# Returns the top left pixel location of the sensed template
def find_target(src, temp):
    '''Uses openCV's matchTemplate to find a desired target inside the source image.

    :param src: Source image
    :param temp: Template image
    :return: Tuple containing the sensed target top left coordinates
    '''

    # Apply template matching
    res = cv.matchTemplate(src, temp, cv.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    return max_loc  # Return top left position


# ---------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------ Statistical Analysis ------------------------------------------------------ #

def evaluate(src_dir, temp_dir, actual_match, n_templates, res_type, resize_value=-1, rotation=0, title=''):
    '''Performs the find_target() function for multiple source images on multiple templates, compares the results with
    the locations in the actual_match list, calculates the displacement and finally; writes the results on a text file.

    :param src_dir: Source images directory
    :param temp_dir: Template images directory
    :param actual_match: List of strings containing the correct coordinates for a give target
    :param n_templates: Number of template images
    :param res_type : Type of result to be return. Either 'text' or 'data'
    :param resize_value: Resize value for process_image()
    :param rotation: Rotation value for process_image()
    :return: A well structured string containing the results of the experiment or a tuple of the resulting data
    '''

    # Append all  the paths into lists
    src_dir = [os.path.join(src_dir, image_path) for image_path in os.listdir(src_dir)]
    sat_paths.sort()
    # Templates paths
    templates_paths = [os.path.join(temp_dir, template_path) for template_path in os.listdir(temp_dir)]
    templates_paths.sort(key=lambda name: int(os.path.splitext(os.path.basename(name))[0]))

    # Read the source images
    src = []
    for src_path in src_dir:
        src.append(cv.cvtColor(cv.imread(src_path)))
    # Read the uav images
    temp = []
    for temp_path in temp_dir:
        temp.append(cv.imread(temp_path))

    # Print all paths to make sure everything is ok
    for path in temp:
        print(path)

    error = float(0)  # Error for the whole set
    error_img = []  # Error for each image Sum(error for every template / number of templates)
    num_of_images = len(src)

    counter = 0  # Keeps track of each iterations
    result = ''

    for img in src:
        img_height, img_width, _ = img.shape
        template_height, template_width, _ = temp[counter].shape

        processed_img = process_image(img)

        print("Source {}x{} , Template {}x{}".format(img_height, img_width, template_height, template_width))

        templates_per_image = n_templates  # This is the num of times the src will be scanned

        error_temp_img = 0  # Error for one template. How displaced the sensed frame is from the actual one

        for i in range(templates_per_image):
            # Store both actual and sensed x and y
            actual_x, actual_y, _ = actual_match[i].split(',')
            sensed_x, sensed_y = find_target(processed_img, process_image(temp[counter], rot_deg=rotation))

            # Error for one template is the added absolute differences between x and y divided the number of pixels
            error_temp_img += np.abs(int(sensed_x) - int(actual_x)) + np.abs(int(sensed_y) - int(actual_y))
            counter += 1

        error_img.append(error_temp_img / templates_per_image)  # Error for a whole image tested with multiple templates
        result += ("Mean error for image {} : {}px\n".format(int(counter / templates_per_image),
                                                             error_temp_img / templates_per_image))

    # Error for all images
    error = sum(error_img) / num_of_images
    result += ("There is {} pixel mean error.\n\n".format(error))

    if res_type == 'text':
        return result
    if res_type == 'data':
        return ['{}'.format(counter + 1) for counter in range(len(error_img))], [round(error_img[counter], 2) for counter
                                                                                 in range(len(error_img))], error, title


# ---------------------------------------------------------------------------------------------------------------------- #

# ----------------------------------------------- Simulation ----------------------------------------------------------- #
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
        print('Satellite image center : {}'.format(sat_image_center))

        # Process the UAV image
        uav_processed_image = cv.cvtColor(sim_uav_images[index], cv.COLOR_BGR2GRAY)

        # Simulate heading errors
        inertial_error = np.random.uniform(0, 2)

        # Rotate the image clockwise
        uav_processed_image = imutils.rotate(uav_processed_image, heading + inertial_error)
        uav_prc_center = (int(uav_processed_image.shape[0] / 2), int(uav_processed_image.shape[1] / 2))
        print('The INS made a {} degree error'.format(inertial_error))

        # Define center of captured image inside the satellite image
        capt_img_rotated_center = (uav_prc_center[0] + dx, uav_prc_center[1] + dy)

        # Doing something wrong here
        x = capt_img_rotated_center[0]
        y = capt_img_rotated_center[1]
        p = uav_prc_center[0]
        q = uav_prc_center[1]
        theta = np.deg2rad(heading)


        actual_capture_coord = (x - p) * np.cos(-theta) + (y - q) * np.sin(-theta) + p, -(x - p) * np.sin(-theta) + (y - q) * np.cos(
            -theta) + q

        # Capturing and rotating image
        capt_top_left = (
            capt_img_rotated_center[0] - int(capture_dim / 2), capt_img_rotated_center[1] - int(capture_dim / 2))
        captured_img = uav_processed_image[capt_top_left[1]:capt_top_left[1] + capture_dim,
                       capt_top_left[0]:capt_top_left[0] + capture_dim]
        cv.imshow('Captured image', captured_img)
        while True:
            if cv.waitKey(1) == 27:
                cv.destroyAllWindows()
                break
        print('INS : {} degrees insertion angle\nRotating image accordingly...'.format(heading))
        captured_img = imutils.rotate(captured_img, -heading)

        print('Captured ground image of size {}x{}'.format(captured_img.shape[0], captured_img.shape[1]))
        captured_img = captured_img[
                       int(captured_img.shape[0] / 4):int(captured_img.shape[0] / 4) + int(captured_img.shape[0] / 2),
                       int(captured_img.shape[1] / 4):int(captured_img.shape[1] / 4) + int(captured_img.shape[1] / 2)]
        cv.imshow('INS corrected captured image', captured_img)
        while True:
            if cv.waitKey(1) == 27:
                cv.destroyAllWindows()
                break

        # Find where the captured image is located relative to the satellite image
        captured_image_location = find_target(sat_images[index], captured_img)  # Top-left location of the template image
        print("DEBUG : captured_image_location={}".format(captured_image_location))

        captured_img_center = (captured_image_location[0] + int(captured_img.shape[0] / 2),
                               captured_image_location[1] + int(captured_img.shape[1] / 2))
        print('Captured image shape {}x{}'.format(captured_img.shape[0], captured_img.shape[1]))

        # Send the course correction signal
        course_displacement = captured_img_center[0] - sat_image_center[0], sat_image_center[1] - captured_img_center[1]

        print('The UAV is off center {} meters horizontally and {} meters vertically\nAnd the error is {:.2f} meters horizontally and {:.2f} meters vertically\n\n'.format(
                course_displacement[0],
                course_displacement[1],
                np.abs(captured_img_center[0] - actual_capture_coord[0]), np.abs(captured_img_center[1] - actual_capture_coord[1])))


# ---------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------- Main --------------------------------------------------------------- #

# Set image directory
sat_directory = '../datasets/sources/source-diverse/1.source'
uav_directory = '../datasets/sources/source-diverse/3.cloudy-images'

# Append all  the paths into lists
sat_paths = [os.path.join(sat_directory, image_path) for image_path in os.listdir(sat_directory)]
sat_paths.sort()

# Append UAV paths
uav_paths = [os.path.join(uav_directory, image_path) for image_path in os.listdir(uav_directory)]
uav_paths.sort()

# Read the satellite images
satellite_images = []
for sat_path in sat_paths:
    satellite_images.append(cv.cvtColor(cv.imread(sat_path), cv.COLOR_BGR2GRAY))
# Read the uav images
uav_images = []
for uav_path in uav_paths:
    uav_images.append(cv.imread(uav_path))

# Print all paths to make sure everything is ok
for elem in uav_paths:
    print(elem)

simulate(satellite_images, uav_images, d_error=100, heading=45)

# --------------------------------- ADD EVERYTHING BELLOW THIS LINE INTO THE EVALUATE METHOD --------------------------- #
# categories = ['Source', 'Blurred', 'Cloudy', 'Blurred and Cloudy']
# results = []
# counter = 0

# # Evaluate all variations of the source imagery
# for directory in source_paths:
#
#     for rot in range(1, 3):
#         file_path = [os.path.join(directory, image_path) for image_path in os.listdir(directory)]
#         file_path.sort()
#
#         # Read the images
#         source_images = []
#         for image_path in file_path:
#             source_images.append(cv.imread(image_path))
#
#         results.append(evaluate(source_images, templates, actual_match_position, 16, 'data', rotation=rot, title='Diverse Dataset {} with {} degree(s) rotation'.format(categories[counter], rot)))
#     counter += 1
#     source_images.clear()

# Evaluate the matching method. The method is hardcoded into the evaluation. This should be changed
# result_text = evaluate(source_images, templates, actual_match_position, 20, rotation=1)

# Write the experiment results on a text file
# file = open("../experiment-results.txt", "a")
# file.write("-------------- Results using 1deg rotation on source images on less-features dataset --------------\n{}".format(result_text))

# Draw plots

# print(results)
#
# colors = ['b', 'g', 'r', 'c', 'm', 'y', '#3277a8', '#a87332', '#915e49']
# counter = 0
# for result in results:
#     fig = plt.figure(counter)
#     plt.bar(result[0], result[1], color=colors[counter])
#     for index, value in enumerate(result[1]):
#         plt.text(index, value, str(value))
#
#     plt.xlabel('Images')
#     plt.ylabel('Mean pixel error')
#     plt.axis([0, 8, 0, 700])
#     ax = plt.gca()
#     ax.set_axisbelow(True)
#     plt.gca().yaxis.grid(linestyle="dashed")
#     plt.show()
#     counter += 1

# ---------------------------------------------------------------------------------------------------------------------- #
