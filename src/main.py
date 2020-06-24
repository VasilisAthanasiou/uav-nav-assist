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
    processed_img = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)

    # Rotate the image
    if rot_deg != 0:
        processed_img = imutils.rotate(processed_img, rot_deg)

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

def evaluate(src_dir, temp_dir, actual_match, n_templates, res_type, resize_value=-1, rotation=0):
    """Performs the find_target() function for multiple source images on multiple templates, compares the results with
    the locations in the actual_match list, calculates the displacement and finally; writes the results on a text file.

    :param src_dir: Source images directory
    :param temp_dir: Template images directory
    :param actual_match: List of strings containing the correct coordinates for a give target
    :param n_templates: Number of template images
    :param res_type : Type of result to be return. Either 'text' or 'data'
    :param resize_value: Resize value for process_image()
    :param rotation: Rotation value for process_image()
    :return: A well structured string containing the results of the experiment or a tuple of the resulting data
    """

    # Append all  the paths into lists
    src_dir = [os.path.join(src_dir, image_path) for image_path in os.listdir(src_dir)]
    src_paths.sort()
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
            sensed_x, sensed_y = findTarget(processed_img, process_image(temp[counter], rot_deg=rotation))

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
                                                                                 in range(len(error_img))], error


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
        capt_top_left = (capt_img_rotated_center[0] - int(capture_dim / 2),
                         capt_img_rotated_center[1] - int(capture_dim / 2))

        captured_img = uav_processed_image[capt_top_left[1]:capt_top_left[1] + capture_dim,
                       capt_top_left[0]:capt_top_left[0] + capture_dim]

        cv.imshow('Captured image', captured_img)
        while True:
            if cv.waitKey(1) == 27:
                cv.destroyAllWindows()
                break
        print('INS : {} degrees insertion angle\nRotating image accordingly...\nPress ESC to continue'.format(heading))
        captured_img = imutils.rotate(captured_img, -heading)

        # Crop the image to get rid of black areas caused by rotation
        captured_img = captured_img[
                       int(captured_img.shape[0] / 4):int(captured_img.shape[0] / 4) + int(captured_img.shape[0] / 2),
                       int(captured_img.shape[1] / 4):int(captured_img.shape[1] / 4) + int(captured_img.shape[1] / 2)]
        cv.imshow('INS corrected captured image', captured_img)
        while True:
            if cv.waitKey(1) == 27:
                cv.destroyAllWindows()
                break

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

def readImages(directory):
    # Append all the paths into lists
    img_paths = [os.path.join(directory, image_path) for image_path in os.listdir(directory)]
    img_paths.sort()

    # Print all paths to make sure everything is ok
    for path in img_paths:
        print(path)

    return [cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2GRAY) for img_path in img_paths]


# ------------------------------------------------------------------------------------------------------------------------ #

# -------------------------------------------------- Main ---------------------------------------------------------------- #

# Set image directory

root_directory = '../datasets/sources/source-diverse'
categories = ['1.source', '2.blurred', '3.cloudy-images', '4.blurred-cloudy']

while True:
    try:
        sat_sel = int(input('Select satellite image source:\n1.source 2.blur 3.cloudy-images 4.blurred-cloudy\n\n'))
        uav_sel = int(input('Select UAV image source:\n1.source 2.blur 3.cloudy-images 4.blurred-cloudy\n\n'))
        head = int(input("Type in the UAV's heading : "))
        dist = int(input("Type in the distance from the center. This will be applied on both axes :\n"))
        break
    except TypeError:
        print('Please select an integer between 1-4')

sat_directory = '../datasets/sources/source-diverse/{}'.format(categories[sat_sel - 1])
uav_directory = '../datasets/sources/source-diverse/{}'.format(categories[uav_sel - 1])

simulate(readImages(sat_directory), readImages(uav_directory), d_error=dist, heading=head)


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
